import numpy as np
import torch
import wandb


def exploration_before_start(args, env_sampler, env_pool, agent, init_exploration_steps=5000):
    # init_exploration_steps = 5000
    for i in range(init_exploration_steps):
        state, action, next_state, reward, done, info = env_sampler.sample(agent, random_explore=True)

        env_pool.push(state, action, reward, next_state, done)


def evaluate_policy(args, env_sampler, agent, epoch_length=1000):
    env_sampler.current_state = None
    env_sampler.path_length = 0

    sum_reward = 0
    sum_cost = 0
    for t in range(epoch_length):
        state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)

        # # extract cost
        # if 'cost' in info:
        #     cost = info['cost']
        # else:
        #     cost = 0

        # use discounted version for cost
        sum_cost += (agent.gamma ** t) * reward[1]
        sum_reward += reward[0]
        if done:
            break

    # reset the environment
    env_sampler.current_state = None
    env_sampler.path_length = 0

    return sum_reward, sum_cost


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    # train_every_n_steps: 1
    if total_step % args.train_every_n_steps > 0:
        return 0
    # max_train_repeat_per_step: 5
    if train_step > args.max_train_repeat_per_step * cur_step:
        return 0

    # num_train_repeat: 20
    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        mmd = np.array([0.])
        model_reward = np.array([0.])
        # Computing MMD
        if args.algo == 'gambol':
            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(model_batch_size))
            # r_max = env_reward.max()

            Y = np.concatenate((env_state, env_action), axis=1)

            env_state = env_state[:env_batch_size]
            env_action = env_action[:env_batch_size]
            env_reward = env_reward[:env_batch_size]
            env_next_state = env_next_state[:env_batch_size]
            env_done = env_done[:env_batch_size]
        else:
            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))

            if args.algo == 'gambol':
                X = np.concatenate((model_state, model_action), axis=1)
                mmd = 2 * agent.gamma * mix_rbf_kernel(X, Y) * args.r_max
                mmd = np.clip(mmd, 0, 10000)
                mmd_penalty = args.penalty * mmd.reshape(mmd.shape[0], 1)
                model_reward = model_reward - mmd_penalty

            model_reward[:, 1] = np.clip(model_reward[:, 1], 0, 1.)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)

        batch_mask = 1 - batch_done
        # batch_done = (~batch_done).astype(int)

        critic_v_loss, critic_cost_loss, min_v, min_c, policy_loss, ent_loss, alpha, lamb = agent.update_parameters(
            (batch_state, batch_action, batch_reward, batch_next_state, batch_mask), args.policy_train_batch_size, i)

        wandb.log({'Training/critic_loss': critic_v_loss,
                   'Training/critic_cost_loss': critic_cost_loss,
                   'Training/policy_loss': policy_loss,
                   'Training/entropy_loss': ent_loss,
                   'Training/alpha': alpha,
                   'Training/lamb': lamb,
                   'Debugging/minV': min_v,
                   'Debugging/minC': min_c})

    return args.num_train_repeat


def mix_rbf_kernel(X, Y, sigma_list=[1, 2, 4, 8, 16]):
    """MMD constraint with Laplacian kernel for support matching"""
    # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

    diff_x_x = X.unsqueeze(1) - X.unsqueeze(0)  # B x N x N x d
    exp_XX = -(diff_x_x.abs()).sum(-1)
    K_XX = 0.0

    diff_x_y = X.unsqueeze(1) - Y.unsqueeze(0)  # B x N x N x d
    exp_XY = -(diff_x_y.abs()).sum(-1)
    K_XY = 0.0

    for sigma in sigma_list:
        K_XX += torch.exp(exp_XX / (2.0 * sigma))
        K_XY += torch.exp(exp_XY / (2.0 * sigma))

    mmd = K_XX.mean(dim=1) - K_XY.mean(dim=1)
    return mmd.detach().numpy()