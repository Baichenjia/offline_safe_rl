import argparse
import os
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

from gym.wrappers import Monitor
import wandb
from sac import SAC, SACLag, CQLLag, ReplayMemory
from cem import ConstrainedCEM

from models import ProbEnsemble, PredictEnv
from imitation import Supervisor, split_data, train_learner
import safety_gym
import env
from batch_utils import *
from mbrl_utils import *
from utils import *

from torch.utils.data import DataLoader

from tqdm import tqdm


def readParser():
    parser = argparse.ArgumentParser(description='BATCH_RL')

    parser.add_argument('--env', default="Safexp-PointGoal1-v0",
        help='Safety Gym environment (default: Safexp-PointGoal1-v0)')
    parser.add_argument('--algo', default="sac",
        help='Must be one of mopo, gambol, sac, cql')
    parser.add_argument('--random_policy', default=False, action='store_true')
    parser.add_argument('--penalize_cost', action='store_true')
    parser.add_argument('--penalty_lambda', type=float, default=1.0)
    parser.add_argument('--tune_penalty', action='store_true')
    parser.add_argument('--colored_noise', action='store_true')
    parser.add_argument('--icem', action='store_true')
    parser.add_argument('--behavioral_cloning', action='store_true')
    parser.add_argument('--use_constraint', dest='feature', action='store_true')
    parser.add_argument('--no_use_constraint', dest='use_constraint', action='store_false')
    parser.set_defaults(use_constraint=True)
    parser.add_argument('--cost_lim', type=float, default=0.,
        help='constraint threshold')
    parser.add_argument('--plan_hor', type=int, default=30)
    # parser.add_argument('--penalty', type=float, default=1.0,
    #     help='reward penalty')
    # parser.add_argument('--cost_penalty', type=float, default=1.0,
    #     help='cost penalty')
    parser.add_argument('--dart_iters', nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40])
    parser.add_argument('--dart', action='store_true')
    parser.add_argument('--cost_size', type=int, default=1,
        help='number of cost functions')
    parser.add_argument('--learn_cost', dest='learn_cost', action='store_true')
    parser.add_argument('--no_learn_cost', dest='learn_cost', action='store_false')
    parser.set_defaults(learn_cost=True)
    # parser.add_argument('--fixed_lamb', dest='fixed_lamb', action='store_true')
    # parser.add_argument('--no_fixed_lamb', dest='fixed_lamb', action='store_false')
    # parser.set_defaults(fixed_lamb=False)
    # parser.add_argument('--lamb', type=float, default=1.0,
    #     help='Lagrangian multiplier')
    # parser.add_argument('--entropy_tuning', dest='entropy_tuning', action='store_true')
    # parser.add_argument('--no_entropy_tuning', dest='entropy_tuning', action='store_false')
    # parser.set_defaults(entropy_tuning=False)
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 0)')

    # parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
    #                 help='size of replay buffer (default: 10000000)')
    parser.add_argument('--model_retain_epochs', type=int, default=20, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
                    help='frequency of training')
    # parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
    #                 help='rollout number M')
    # parser.add_argument('--rollout_length', type=int, default=1, metavar='A',
    #                     help='rollout length')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')

    parser.add_argument('--num_epoch', type=int, default=200, metavar='A',
                    help='total number of epochs')
    # parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
    #                 help='minimum pool size')
    # parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
    #                 help='ratio of env samples / model samples')
    # parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
    #                 help='initial random exploration steps')
    # parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
    #                 help='frequency of training policy')
    # parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
    #                 help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    # parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
    #                 help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')

    parser.add_argument('--hidden_size', type=int, default=200, metavar='A',
                    help='ensemble model hidden dimension')
    # parser.add_argument('--model_type', default='pytorch', metavar='A',
    #                 help='predict model -- pytorch or tensorflow')
    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, cem_agent, agent, env_pool, expert_pool):
    reward_sum = 0
    environment_step = 0
    learner_update_step = 0
    eps_idx = 0
    env = env_sampler.env

    for epoch_step in tqdm(range(args.num_epoch)):
        monitor = Monitor(env, f"videos/{args.run_name}", force=True)
        if epoch_step % 10 == 0:
            env_sampler.env = monitor
            monitor.render()

        epoch_rewards = [0]
        epoch_costs = [0]
        epoch_lens = [0]
        epoch_logout = {}

        if (epoch_step+1) % 100 == 0:
            agent_path = f'saved_policies/{args.env}-{args.run_name}-epoch{epoch_step+1}'
            agent.save_model(agent_path)

        for i in range(args.epoch_length):
            # plan CEM action
            wandb.log({
                "Train/episode_number": eps_idx,
                "Train/episode_step": environment_step,
                })
            cur_state, action, next_state, reward, done, info = env_sampler.sample(cem_agent)
            epoch_rewards[-1] += reward[0]
            epoch_costs[-1] += args.c_gamma ** i * reward[1]
            epoch_lens[-1] += 1

            env_pool.push(cur_state, action, reward, next_state, done)
            if cem_agent.model != None:
                expert_pool.push(cur_state, action, reward, next_state, done)

            environment_step += 1

            if done and i != args.epoch_length - 1:
                epoch_rewards.append(0)
                epoch_costs.append(0)
                epoch_lens.append(0)
                eps_idx += 1

            if (i + 1) % args.model_train_freq == 0:
                assert(args.algo not in ['sac', 'cql'])
                train_predict_model(args, env_pool, predict_env)
                if not args.random_policy:
                    cem_agent.set_model(predict_env.model)

        epoch_reward = np.mean(epoch_rewards)
        epoch_cost = np.mean(epoch_costs)
        epoch_len = np.mean(epoch_lens)
        print("")
        print(f'Epoch {epoch_step} Train_Reward {epoch_reward:.2f} Train_Cost {epoch_cost:.2f} Train_Len {epoch_len:.2f}')

        monitor.close()
        env_sampler.env = env

        if args.tune_penalty:
            cem_agent.optimize_penalty_lambda(epoch_cost)

        if args.behavioral_cloning and len(expert_pool):
            dataloader = DataLoader(expert_pool, batch_size=args.policy_train_batch_size, shuffle=True)
            for expert_states, expert_actions, _, _, _ in dataloader:
                expert_states = expert_states.float().to(agent.device)
                expert_actions = expert_actions.float().to(agent.device)
                agent.policy_optim.zero_grad()
                policy_loss = -agent.policy.log_prob(expert_states, expert_actions).mean()
                policy_loss.backward()
                agent.policy_optim.step()
                loss_val = policy_loss.cpu().detach().item()
                wandb.log({
                    "Policy/update_step": learner_update_step,
                    "Policy/loss": loss_val
                    })
                learner_update_step += 1

            rewards = [evaluate_policy(args, env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]
            rewards = np.array(rewards)

            rewards_avg = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
            
            print(f'Epoch {epoch_step} Eval_Reward {rewards_avg[0]:.2f} Eval_Cost {rewards_avg[1]:.2f}')
            epoch_logout.update({
                'Eval/eval_reward': rewards_avg[0],
                'Eval/eval_cost': rewards_avg[1],
                'Eval/reward_std': rewards_std[0],
                'Eval/cost_std': rewards_std[1],
            })

        wandb.log({
            'Train/epoch':epoch_step,
            'Train/env_step': environment_step,
            'Train/epoch_reward': epoch_reward,
            'Train/epoch_cost': epoch_cost,
            'Train/epoch_length': epoch_len,
            **epoch_logout,
        })


def learn(args, teacher, learner, env_sampler):
    train_trajs = []
    holdout_trajs = []
    act_dim = np.prod(env_sampler.env.action_space.shape)
    supervisor = Supervisor(act_dim, teacher, learner)
    for itr in tqdm(range(args.dart_iters[-1])):
        epoch_reward = 0
        epoch_cost = 0
        traj = []
        for i in range(args.epoch_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(supervisor)
            i_action = supervisor.i_action
            epoch_reward += reward[0]
            epoch_cost += reward[1] * args.gamma ** i
            traj.append((cur_state, action, i_action))
            if done:
                break
        train_traj, holdout_traj = split_data(traj)
        train_trajs.append(train_traj) 
        holdout_trajs.append(holdout_traj)
        
        wandb.log({
            "Dart/supervisor_reward": epoch_reward,
            "Dart/supervisor_cost": epoch_cost,
        })
        
        if itr + 1 in args.dart_iters:
            train_learner(args, learner, train_trajs, env_sampler)
            supervisor.fit_cov(holdout_trajs[-5:])

def main():
    args = readParser()
    if not args.use_constraint:
        spec = 'noconstraint'
    elif args.use_constraint:
        if args.penalize_cost:
            spec = f'P{args.penalty_lambda}-C{args.cost_lim}'
        else:
            spec = f'C{args.cost_lim}'

    run_name = f"{args.algo}-{spec}-{args.seed}"
    args.run_name = run_name

    # Initial environment
    env = gym.make(args.env)

    args.gamma = 0.99
    args.c_gamma = 0.99
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    # env.action_space.seed(args.seed)

    # initialize policy agent and cem optimizer
    agent = SAC(env.observation_space.shape[0], env.action_space,
                       gamma=args.gamma)

    # use all batch data for model-free methods
    # if args.algo in ['sac', 'cql']:
    #     args.real_ratio = 1.0
    #     args.num_epoch = 1000
    #     args.num_train_repeat = 1

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    cost_size = args.cost_size

    # initialize dynamics model
    env_model = ProbEnsemble(state_size, action_size, network_size=5,
                             reward_size=1+cost_size, hidden_size=args.hidden_size)
    if args.cuda:
        env_model.to('cuda')

    # Imaginary Environment
    predict_env = PredictEnv(env_model, args.env)

    cem_agent = ConstrainedCEM(env,
                               plan_hor=args.plan_hor,
                               gamma=args.gamma,
                               cost_lim=args.cost_lim,
                               use_constraint=args.use_constraint,
                               use_colored_noise=args.colored_noise,
                               use_icem=args.icem,
                               penalize_cost=args.penalize_cost,
                               tune_penalty=args.tune_penalty,
                               penalty_lambda=args.penalty_lambda,
                               termination_function = predict_env.termination_fn
                               )

    # only can set tune_penalty to true if CCEM is penalizing cost
    if args.tune_penalty:
        assert args.penalize_cost
    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

    # Initial pool for env
    env_pool = ReplayMemory(args.epoch_length * args.num_epoch)

    # Initial pool for model
    model_pool = ReplayMemory(args.epoch_length * args.model_retain_epochs)

    wandb.init(project='safety-gym',
               group=args.env,
               name=run_name,
               config=args,
               monitor_gym=True
               )

    # Train
    train(args, env_sampler, predict_env, cem_agent, agent, env_pool, model_pool)
    del env_pool
    del model_pool

    if args.dart:
        learn(args, cem_agent, agent, env_sampler)


if __name__ == '__main__':
    main()