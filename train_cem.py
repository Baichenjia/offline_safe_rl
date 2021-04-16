import argparse
import os
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

import wandb
from sac import SAC, SACLag, CQLLag, ReplayMemory
from cem import ConstrainedCEM

from models import ProbEnsemble, PredictEnv
import safety_gym
import env
from batch_utils import *
from mbrl_utils import *
from utils import *

from tqdm import tqdm


def readParser():
    parser = argparse.ArgumentParser(description='BATCH_RL')

    parser.add_argument('--env', default="Safexp-PointGoal1-v0",
        help='Safety Gym environment (default: Safexp-PointGoal1-v0)')
    parser.add_argument('--algo', default="sac",
        help='Must be one of mopo, gambol, sac, cql')

    # Lagrangian + MBRL hyperparameters
    parser.add_argument('--use_constraint', dest='feature', action='store_true')
    parser.add_argument('--no_use_constraint', dest='use_constraint', action='store_false')
    parser.set_defaults(use_constraint=True)
    parser.add_argument('--cost_lim', type=float, default=0.,
        help='constraint threshold')
    parser.add_argument('--penalty', type=float, default=1.0,
        help='reward penalty')
    parser.add_argument('--cost_penalty', type=float, default=1.0,
        help='cost penalty')
    parser.add_argument('--cost_size', type=int, default=1,
        help='number of cost functions')
    parser.add_argument('--learn_cost', dest='learn_cost', action='store_true')
    parser.add_argument('--no_learn_cost', dest='learn_cost', action='store_false')
    parser.set_defaults(learn_cost=True)
    parser.add_argument('--fixed_lamb', dest='fixed_lamb', action='store_true')
    parser.add_argument('--no_fixed_lamb', dest='fixed_lamb', action='store_false')
    parser.set_defaults(fixed_lamb=False)
    parser.add_argument('--lamb', type=float, default=1.0,
        help='Lagrangian multiplier')
    parser.add_argument('--entropy_tuning', dest='entropy_tuning', action='store_true')
    parser.add_argument('--no_entropy_tuning', dest='entropy_tuning', action='store_false')
    parser.set_defaults(entropy_tuning=False)
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 0)')

    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--model_retain_epochs', type=int, default=5, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                    help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                    help='rollout number M')
    parser.add_argument('--rollout_length', type=int, default=1, metavar='A',
                        help='rollout length')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')

    parser.add_argument('--num_epoch', type=int, default=200, metavar='A',
                    help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                    help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                    help='ratio of env samples / model samples')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                    help='initial random exploration steps')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                    help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                    help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')

    parser.add_argument('--hidden_size', type=int, default=200, metavar='A',
                    help='ensemble model hidden dimension')
    parser.add_argument('--model_type', default='pytorch', metavar='A',
                    help='predict model -- pytorch or tensorflow')
    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, cem_agent, agent, env_pool, expert_pool):
    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_length
    exploration_before_start(args, env_sampler, env_pool, cem_agent)

    for epoch_step in tqdm(range(args.num_epoch)):
        # if (epoch_step+1) % 100 == 0:
        #     agent_path = f'saved_policies/{args.env}-{args.run_name}-epoch{epoch_step+1}'
        #     agent.save_model(agent_path)

        start_step = total_step
        train_policy_steps = 0
        epoch_reward = 0
        epoch_cost = 0
        for i in range(args.epoch_length):
            cur_step = total_step - start_step

            # epoch_length = 1000, min_pool_size = 1000
            if cur_step >= args.epoch_length and len(env_pool) > 1000:
                break

            if cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                assert(args.algo not in ['sac', 'cql'])
                train_predict_model(args, env_pool, predict_env)
                cem_agent.set_model(predict_env.model)

            # plan CEM action
            cur_state, action, next_state, reward, done, info = env_sampler.sample(cem_agent)
            env_pool.push(cur_state, action, reward, next_state, done)
            expert_pool.push(cur_state, action, reward, next_state, done)

            epoch_reward += reward[0]
            epoch_cost += reward[1]
            total_step += 1

        for _ in range(100):
            expert_states, expert_actions, _, _, _ = expert_pool.sample(args.policy_train_batch_size)
            expert_states = torch.FloatTensor(expert_states).to(agent.device)
            expert_actions = torch.FloatTensor(expert_actions).to(agent.device)
            # mean, _ = agent.policy(expert_states)
            # policy_loss = torch.nn.MSELoss()(mean, expert_actions)
            agent.policy_optim.zero_grad()
            policy_loss = -agent.policy.log_prob(expert_states, expert_actions).mean()
            policy_loss.backward()
            agent.policy_optim.step()
            loss_val = policy_loss.cpu().detach().item()
            wandb.log({"Policy/loss": loss_val})

        rewards = [evaluate_policy(args, env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]
        rewards = np.array(rewards)

        rewards_avg = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        print("")
        print(f'Epoch {epoch_step} Train_Reward {epoch_reward:.2f} Train_Cost {epoch_cost:.2f}')
        print(f'Epoch {epoch_step} Eval_Reward {rewards_avg[0]:.2f} Eval_Cost {rewards_avg[1]:.2f}')
        wandb.log({'Eval/epoch':epoch_step,
                   'Train/epoch_reward': epoch_reward,
                   'Train/epoch_cost': epoch_cost,
                   'Eval/eval_reward': rewards_avg[0],
                   'Eval/eval_cost': rewards_avg[1],
                   'Eval/reward_std': rewards_std[0],
                   'Eval/cost_std': rewards_std[1]})


def main():
    args = readParser()
    if not args.use_constraint:
        spec = 'noconstraint'
    elif args.use_constraint:
        if args.fixed_lamb:
            spec = f'lambda{args.lamb}-C{args.cost_lim}'
        elif not args.fixed_lamb:
            spec = f'lagrangian{args.lamb}-C{args.cost_lim}'

    run_name = f"{args.algo}-{spec}-{args.seed}"
    args.run_name = run_name
    # from ipdb import set_trace
    # set_trace()
    # Initial environment
    env = gym.make(args.env)

    if 'Circle' in args.env:
        args.gamma = 0.995
    else:
        args.gamma = 0.99

    if 'Pendulum' in args.env:
        args.epoch_length = 100
        args.model_train_freq = 100
        args.num_epoch = 100
    elif 'PointMass' in args.env:
        args.epoch_length = 200
        args.model_train_freq = 200
        args.num_epoch = 100

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # initialize policy agent and cem optimizer
    agent = SAC(env.observation_space.shape[0], env.action_space,
                       gamma=args.gamma)
    cem_agent = ConstrainedCEM(env, gamma=args.gamma, use_constraint=args.use_constraint)

    # use all batch data for model-free methods
    if args.algo in ['sac', 'cql']:
        args.real_ratio = 1.0
        args.num_epoch = 1000
        args.num_train_repeat = 1

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    cost_size = args.cost_size

    # determine if the ensemble should learn the cost
    if not args.learn_cost:
        cost_size = 0

    # initialize dynamics model
    env_model = ProbEnsemble(state_size, action_size, network_size=5,
                             reward_size=1+cost_size, hidden_size=args.hidden_size)
    if args.cuda:
        env_model.to('cuda')

    # Imaginary Environment
    predict_env = PredictEnv(env_model, args.env)

    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)

    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(args.rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    wandb.init(project='safety-gym',
               group=args.env,
               name=run_name,
               config=args)

    # Train
    train(args, env_sampler, predict_env, cem_agent, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()