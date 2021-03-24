import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import env
from sac import SACLag


def evaluate_policy(env, agent, num_eisodes=30, epoch_length=100):
    for episode in range(num_eisodes):
        state = env.reset()

        ep_reward = 0
        ep_cost = 0
        for t in range(epoch_length):
            action = agent.select_action(state, eval=True)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_cost += (0.995 ** t) * info['cost']
            # print(t)
            # env.render()
            state = next_state
            if done:
                break
        print(f'Episode {episode} \t ep_reward {ep_reward} \t ep_cost {ep_cost}')


def readParser():
    parser = argparse.ArgumentParser(description='cross')
    parser.add_argument('--env', default="AntCircle-v0",
        help='Gym environment (default: AntCircle-v0)')
    parser.add_argument('--algo', default="sac",
        help='RL algorithm (default: sac)')
    return parser.parse_args()


def main():
    args = readParser()
    args.env = 'AntCircle-v0'
    env = gym.make(args.env)

    env.seed(0)
    args.epoch_length = 1000
    agent = SACLag(env.observation_space.shape[0], env.action_space)

    epoch = 1000
    algo = 'sac'

    spec = 'lagrangian1.0-C50.0-0'
    # spec = 'noconstraint-0'
    path = f'{args.env}-{algo}-{spec}-epoch{epoch}'

    agent.load_model('saved_policies/' + path)
    evaluate_policy(env, agent, epoch_length=args.epoch_length)



if __name__ == '__main__':
    main()