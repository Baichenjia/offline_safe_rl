from config import default
import torch
import numpy as np

cost_function = default.cost_function
termination_function = default.termination_function

def reward_function(obs, act):
    velocity_reward = obs[:, 5]  # the qvel for the root-x joint
    height_cost = -3 * (obs[:, 0] - 1.3) ** 2  # the height
    obs_reward = velocity_reward + height_cost

    if isinstance(act, np.ndarray):
        act_cost = -0.1 * np.sum(act ** 2, axis=1)
    else:
        act_cost = -0.1 * torch.sum(act ** 2, axis=1)
    return obs_reward + act_cost