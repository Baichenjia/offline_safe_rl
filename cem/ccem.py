import math
import numpy as np
import colorednoise as cn
import torch
from torch.optim import Adam

import scipy.stats as stats
import wandb


class ConstrainedCEM:
    def __init__(self,
                 env,
                 termination_function=None,
                 epoch_length=1000,
                 plan_hor=30,
                 gamma=0.99,
                 c_gamma=0.99,
                 penalty_lambda=1,
                 noise_beta=0.25,
                 cost_lim=5,
                 tune_penalty=False,
                 use_constraint=True,
                 use_icem=False,
                 use_colored_noise=False,
                 penalize_cost=True,
                 ):
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.gamma = gamma
        self.c_gamma = c_gamma

        self.tune_penalty = tune_penalty
        if not self.tune_penalty:
            self.penalty_lambda = torch.tensor([penalty_lambda], requires_grad=False)
        else:
            self.log_penalty_lambda = torch.tensor([math.log(penalty_lambda)], requires_grad=True)
            self.lamb_optim = Adam([self.log_penalty_lambda], lr=0.01)
            self.penalty_lambda = self.log_penalty_lambda.exp()

        self.termination_function = termination_function
        self.noise_beta = noise_beta
        self.cost_lim = cost_lim
        self.epoch_length = epoch_length
        self.use_constraint = use_constraint
        self.use_icem = use_icem
        self.use_colored_noise = use_icem or use_colored_noise
        self.penalize_cost = penalize_cost
        self.device = 'cuda:0'

        # cem optimization hyperparams
        self.per = 1
        self.npart = 20
        self.plan_hor = plan_hor # Same
        self.popsize = 500 # Same
        self.num_elites = 50 # Same
        self.max_iters = 5 # Same
        self.alpha = 0.1 # Same
        self.epsilon = 0.001
        self.lb = np.tile(self.ac_lb, [self.plan_hor])
        self.ub = np.tile(self.ac_ub, [self.plan_hor])
        self.decay = 1.25
        self.elite_fraction = 0.3
        self.elites = None

        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        self.model = None
        self.step = 0
        self.log_period = epoch_length / 25
        # self.log_period = 1

    def set_model(self, model):
        self.model = model

    def select_action(self, obs, eval_t=False):
        if self.model is None:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        soln = self.obtain_solution(obs, self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.select_action(obs)

    def obtain_solution(self, obs, init_mean, init_var):
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            if self.use_icem:
                popsize = int(max(2 * self.num_elites, self.popsize * self.decay ** -t))
            else:
                popsize = self.popsize

            if self.use_colored_noise:
                noise = cn.powerlaw_psd_gaussian(self.noise_beta, [popsize, self.plan_hor * self.dU])
            else:
                noise = X.rvs(size=[popsize, self.plan_hor * self.dU])

            samples = noise * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            
            if self.use_icem and self.elites is not None:
                if t == 0:
                    random_actions = (X.rvs(size=[self.num_elites, self.plan_hor * self.dU]) * np.sqrt(constrained_var) + mean)[:, -1:]
                    self.elites = np.concatenate([self.elites[:, 1:], random_actions], axis=1)
                elite_samples_id = np.random.choice(np.arange(self.num_elites), int(self.num_elites * self.elite_fraction))
                elite_samples = self.elites[elite_samples_id]
                samples = np.concatenate([samples, elite_samples])

            rewards, costs, eps_lens = self.rollout(obs, samples)
            epoch_ratio = np.ones_like(eps_lens) * self.epoch_length / self.plan_hor
            terminated = eps_lens != self.plan_hor
            # epoch_ratio = terminated + ~terminated * epoch_ratio
            c_gamma_discount = (1 - self.c_gamma ** (epoch_ratio * self.plan_hor)) / (1 - self.c_gamma) / self.plan_hor
            rewards = rewards * epoch_ratio
            costs = costs * c_gamma_discount

            feasible_ids = ((costs < self.cost_lim) & (~terminated)).nonzero()[0]
            if self.use_constraint:
                if feasible_ids.shape[0] >= self.num_elites:
                    elite_ids = feasible_ids[np.argsort(-rewards[feasible_ids])][:self.num_elites]
                else:
                    elite_ids = np.argsort(costs)[:self.num_elites]
            else:
                elite_ids = np.argsort(-rewards)[:self.num_elites]
            self.elites = samples[elite_ids]
            new_mean = np.mean(self.elites, axis=0)
            new_var = np.var(self.elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            average_reward = rewards.mean().item()
            average_cost = costs.mean().item()
            average_len = eps_lens.mean().item()
            average_elite_reward = rewards[elite_ids].mean().item()
            average_elite_cost = costs[elite_ids].mean().item()
            average_elite_len = eps_lens[elite_ids].mean().item()
            if t == 0:
                start_reward = average_reward
                start_cost = average_cost
            t += 1
        
        if self.step % self.log_period == 0:
            wandb.log({ "CEM/Step": self.step,
                        "CEM/PenaltyLambda": self.penalty_lambda,
                        "CEM/AverageReward": average_reward,
                        "CEM/AverageCost": average_cost,
                        "CEM/AverageLen": average_len,
                        "CEM/EliteReward": average_elite_reward,
                        "CEM/EliteCost": average_elite_cost,
                        "CEM/EliteLen": average_elite_len,
                        "CEM/DeltaReward": average_reward - start_reward,
                        "CEM/DeltaCost": average_cost - start_cost,
                        "CEM/FeasibleSamples": feasible_ids.shape[0],
                        })
        
        self.step += 1
        if self.use_icem:
            return self.elites[0]
        else:
            return mean

    @torch.no_grad()
    def rollout(self, obs, ac_seqs):
        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.device)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 1)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 1)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 1)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 1)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 1)

        # Expand current observation
        cur_obs = torch.from_numpy(obs).float().to(self.device)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        rewards = torch.zeros(nopt, self.npart, device=self.device)
        costs = torch.zeros(nopt, self.npart, device=self.device)
        dones = torch.zeros(nopt, self.npart, dtype=bool, device=self.device)
        length = torch.zeros(nopt, self.npart, device=self.device)

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            obs_delta, reward, cost, done = self._predict_next(cur_obs, cur_acs)
            done = done.view(-1, self.npart)
            reward = reward.view(-1, self.npart)
            cost = cost.view(-1, self.npart)

            dones = dones | done
            rewards += reward * ~dones
            termination_penalty = cost.mean()
            costs += cost * ~dones
            costs += termination_penalty * dones
            length += ~dones
            cur_obs = cur_obs + obs_delta

            if t == 0:
                start_reward = reward
                start_cost = cost

        if self.step % self.log_period == 0:
            wandb.log({
                    "ModelRollout/StartReward": start_reward.mean(),
                    "ModelRollout/StartStdReward": start_reward.std(),
                    "ModelRollout/StartCost": start_cost.mean(),
                    "ModelRollout/StartStdCost": start_cost.std(),
                    "ModelRollout/EndReward": reward.mean(),
                    "ModelRollout/EndStdReward": reward.std(),
                    "ModelRollout/EndCost": cost.mean(),
                    "ModelRollout/EndStdCost": cost.std(),
                    "ModelRollout/EndDone": dones.float().mean(),
                    })


        # Replace nan with high cost
        rewards[rewards != rewards] = -1e6
        costs[costs != costs] = 1e6

        # TODO: both rewards and costs should be returned
        return rewards.mean(dim=1).detach().cpu().numpy(), costs.mean(dim=1).detach().cpu().numpy(), length.mean(dim=1).detach().cpu().numpy()

    def optimize_penalty_lambda(self, epoch_cost):
        lamb_loss = -(self.log_penalty_lambda * (epoch_cost - self.cost_lim))

        self.lamb_optim.zero_grad()
        lamb_loss.backward()
        self.lamb_optim.step()

        self.penalty_lambda = self.log_penalty_lambda.exp().detach()

    def _predict_next(self, obs, acs):
        # print(obs.shape, acs.shape)
        proc_obs = self._expand_to_ts_format(obs)
        proc_acs = self._expand_to_ts_format(acs)
        # print(proc_obs.shape, acs.shape)

        inputs = torch.cat((proc_obs, proc_acs), dim=-1)

        mean, var = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=self.device)* var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        rewards, obs_delta = predictions[:, :self.model.reward_size], predictions[:, self.model.reward_size:]

        reward = rewards[:, 0]

        done = self.termination_function(
            obs.detach().cpu().numpy(),
            acs.detach().cpu().numpy(),
            (obs + obs_delta).detach().cpu().numpy(),
            )
        done = torch.from_numpy(done).to(self.device)

        if not self.use_constraint and rewards.shape[1] < 2:
            cost = torch.zeros_like(reward)
        else:
            cost = rewards[:, 1]
        
        if self.use_constraint and self.penalize_cost:
            cost_penalty = var.sqrt().norm(dim=2).max(0)[0].repeat_interleave(self.model.num_nets)
            cost += self.penalty_lambda.to(cost_penalty.device) * cost_penalty

        return obs_delta, reward, cost, done

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped


if __name__ == '__main__':
    import gym
    import env
    from models.ensemble import ProbEnsemble

    env_name = 'AntTruncated-v0'
    env = gym.make(env_name)
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    env_model = ProbEnsemble(state_size, action_size, network_size=5, reward_size=2, hidden_size=30)
    env_model.to('cuda:0')

    optimizer = ConstrainedCEM(env)
    optimizer.set_model(env_model)

    state = env.reset()
    action = optimizer.act(state)

