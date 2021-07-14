import env
import gym
import importlib
import numpy as np

def reward_function(obs, act):
    return None

def cost_function(obs, act):
    return None

def termination_function(obs, act, next_obs):
    done = np.array([False]).repeat(len(obs))
    done = done[:, None]
    return done

class default_config:
    reward_function = reward_function
    cost_function = cost_function
    termination_function = termination_function

def create_config(env_name):
    try:
        config = importlib.import_module(f"config.{env_name.lower()}")
    except ImportError:
        config = default_config
    config.env = gym.make(env_name)

    return config