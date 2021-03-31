import numpy as np
import torch
import scipy.stats as stats


class ConstrainedCEM:
    def __init__(self, env, gamma=0.99, cost_lim=5, use_constraint=True):
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.gamma = gamma
        self.cost_lim = cost_lim
        self.use_constraint = use_constraint

        # cem optimization hyperparams
        self.per = 1
        self.npart = 20
        self.plan_hor = 30
        self.popsize = 500
        self.num_elites = 50
        self.max_iters = 5
        self.alpha = 0.1
        self.epsilon = 0.001
        self.lb = np.tile(self.ac_lb, [self.plan_hor])
        self.ub = np.tile(self.ac_ub, [self.plan_hor])

        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        self.model = None
        self.device = 'cuda:0'

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

            samples = X.rvs(size=[self.popsize, self.plan_hor * self.dU]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            rewards, costs = self.rollout(obs, samples)

            feasible_ids = (costs < self.cost_lim).nonzero()[0]
            if self.use_constraint:
                if feasible_ids.shape[0] >= self.num_elites:
                    elite_ids = feasible_ids[np.argsort(rewards[feasible_ids])]
                    elites = samples[elite_ids][self.num_elites:]
                else:
                    elites = samples[np.argsort(costs)][:self.num_elites]
            else:
                elites = samples[np.argsort(rewards)][self.num_elites:]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

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

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            next_obs, reward, cost = self._predict_next(cur_obs, cur_acs)
            # from ipdb import set_trace
            # set_trace()
            reward = reward.view(-1, self.npart)
            cost = cost.view(-1, self.npart)

            rewards += reward
            costs += cost
            cur_obs = cur_obs + next_obs

        # Replace nan with high cost
        rewards[rewards != rewards] = -1e6
        costs[costs != costs] = 1e6

        # TODO: both rewards and costs should be returned
        return rewards.mean(dim=1).detach().cpu().numpy(), costs.mean(dim=1).detach().cpu().numpy()

    def _predict_next(self, obs, acs):
        # print(obs.shape, acs.shape)
        proc_obs = self._expand_to_ts_format(obs)
        acs = self._expand_to_ts_format(acs)
        # print(proc_obs.shape, acs.shape)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=self.device) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        rewards, next_obs = predictions[:, :self.model.reward_size], predictions[:, self.model.reward_size:]

        return next_obs, rewards[:, 0], rewards[:, 1]

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

