import argparse
import os
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

import wandb
from sac import SAC, CQL, ReplayMemory
from models import ProbEnsemble, PredictEnv
from batch_utils import *
from mbrl_utils import *
from utils import *

from tqdm import tqdm


def readParser():
    parser = argparse.ArgumentParser(description='BATCH_RL')

    parser.add_argument('--env', default="halfcheetah-medium-replay-v0",
        help='Mujoco Gym environment (default: halfcheetah-medium-replay-v0)')
    parser.add_argument('--algo', default="mopo",
        help='Must be one of mopo, gambol, sac, cql')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.set_defaults(pretrained=False)
    
    parser.add_argument('--penalty', type=float, default=1.0,
        help='reward penalty')

    parser.add_argument('--rollout_length', type=int, default=5, metavar='A',
                        help='rollout length')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 0)')

    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--model_retain_epochs', type=int, default=5, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
                    help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=50000, metavar='A',
                    help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')

    parser.add_argument('--num_epoch', type=int, default=500, metavar='A',
                    help='total number of epochs')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                    help='ratio of env samples / model samples')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                    help='initial random exploration steps')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                    help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=1, metavar='A',
                    help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                    help='predict model -- pytorch or tensorflow')
    parser.add_argument('--pre_trained', type=bool, default=False,
                    help='flag for whether dynamics model pre-trained')
    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_length

    for epoch_step in tqdm(range(args.num_epoch)):
        if (epoch_step+1) % 100 == 0:
            agent_path = f'saved_policies/{args.env}-{args.run_name}-epoch{epoch_step+1}'
            agent.save_model(agent_path)

        start_step = total_step
        train_policy_steps = 0
        for i in range(args.epoch_length):
            cur_step = total_step - start_step

            # epoch_length = 1000, min_pool_size = 1000
            if cur_step >= args.epoch_length:
                break

            if cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                assert(args.algo not in ['sac', 'cql'])
                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

            # train policy
            train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
            total_step += 1

        rewards = [evaluate_policy(args, env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]
        rewards = np.array(rewards)
        rewards_avg = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)

        min_score = REF_MIN_SCORE[args.env]
        max_score = REF_MAX_SCORE[args.env]
        normalized_score = 100 * (rewards_avg - min_score) / (max_score - min_score)

        print("")
        print(f'Epoch {epoch_step} Eval_Reward {rewards_avg:.2f} Normalized_Score {normalized_score:.2f}')
        wandb.log({'Eval/epoch':epoch_step,
                   'Eval/reward_mean': rewards_avg,
                   'Eval/normalized_score': normalized_score,
                   'Eval/reward_std': rewards_std})


def main():
    args = readParser()

    run_name = f"{args.algo}-{args.seed}"

    # Initial environment
    env, dataset = load_d4rl_dataset(args.env)

    # use all batch data for model-free methods
    if args.algo in ['sac', 'cql']:
        args.real_ratio = 1.0

    wandb.init(project='batch_rl',
               group=args.env,
               name=run_name,
               config=args)
    args.run_name = run_name

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Initial agent (SAC or WCSAC)
    if args.algo != 'cql':
        agent = SAC(env.observation_space.shape[0], env.action_space)
    else:
        from sac.cql import CQL
        agent = CQL(env.observation_space.shape[0], env.action_space)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)


    # initialize dynamics model
    env_model = ProbEnsemble(state_size, action_size, reward_size=1)
    if args.cuda:
        env_model.to('cuda')

    # load pre-trained dynamics model
    model_path = f'saved_models/{args.env}-ensemble.pt'

    if os.path.exists(model_path):
        env_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        args.pre_trained = True
        env_model.fit_input = True
        print("Use Pre-trained Single-step Model!")
        env_model.elite_model_idxes = [i for i in range(env_model.num_nets)]

    # Imaginary Environment
    predict_env = PredictEnv(env_model, args.env)

    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

    # Initial replay buffer for env
    if dataset is not None:
        n = dataset['observations'].shape[0]
        print(f"{args.env} dataset size {n}")
        env_pool = ReplayMemory(n)
        for i in range(n):
            state, action, reward, next_state, done = dataset['observations'][i], dataset['actions'][i], dataset['rewards'][
                i], dataset['next_observations'][i], dataset['terminals'][i]
            env_pool.push(state, action, reward, next_state, done)
    else:
        env_pool = ReplayMemory(args.init_exploration_steps)
        exploration_before_start(args, env_sampler, env_pool, agent, init_exploration_steps=args.init_exploration_steps)

    # rmax
    env_pool_rewards = env_pool.return_all()[2]
    global r_max
    r_max = np.max(env_pool_rewards)

    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(args.rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # train dynamics model if not pre-trained
    if not args.pre_trained and args.algo not in ['sac', 'cql']:
        print("Training predictive model!")
        train_predict_model(args, env_pool, predict_env)

    # Train
    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()