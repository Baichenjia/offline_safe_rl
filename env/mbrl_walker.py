import gym
from gym import Wrapper

class MBRLWalker(Wrapper):
    def __init__(self,
                 name,
                 ctrl_coeff=0.1,
                 height_coeff=3,
                 alive_bonus=1,
                 target_height=1.3,
                 velocity_idx=5,
                 height_idx=0):
        env = gym.make(name)
        super(MBRLWalker, self).__init__(env)
        self.ctrl_coeff = ctrl_coeff
        self.height_coeff = height_coeff
        self.alive_bonus = alive_bonus
        self.target_height = target_height
        self.velocity_idx=velocity_idx
        self.height_idx = height_idx

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward_velocity = observation[self.velocity_idx]

        reward_control = - self.ctrl_coeff * (action ** 2).sum()

        reward_height = - self.height_coeff * (observation[self.height_idx] - self.target_height) ** 2

        reward = reward_velocity + reward_height + reward_control + self.alive_bonus

        done = False

        return observation, reward, done, info

