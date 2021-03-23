import numpy as np
import torch
import wandb

def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))

    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)

    reward = np.reshape(reward, (reward.shape[0], -1))
    labels = np.concatenate((reward, delta_state), axis=-1)

    val_mse, val_nll = predict_env.model.train(inputs, labels, batch_size=256)
    wandb.log({'Model/model_nll': val_nll,
               'Model/model_rmse': val_mse})
    # save trained dynamics model
    if args.learn_cost:
        model_path = f'saved_models/{args.env}-ensemble-h{args.hidden_size}.pt'
    else:
        model_path = f'saved_models/{args.env}-ensemble-nocost-h{args.hidden_size}.pt'
    torch.save(predict_env.model.state_dict(), model_path)


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    # rollout_batch_size = 50000
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)

    for i in range(rollout_length):
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action, reward_penalty=args.penalty,
                                                                 cost_penalty=args.cost_penalty, algo=args.algo)
        if not args.learn_cost:
            # TODO: insert custom safe-gym cost function
            raise NotImplementedError

        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False, random_explore=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        if not random_explore:
            action = agent.select_action(self.current_state, eval_t)
        else:
            action = self.env.action_space.sample()

        next_state, reward, terminal, info = self.env.step(action)
        # if eval_t:
        #     self.env.render()
        self.path_length += 1
        self.sum_reward += reward

        # add the cost
        assert('cost' in info)
        cost = info['cost']
        reward = np.array([reward, cost])

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info