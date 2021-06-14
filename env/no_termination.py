import gym

class NoTermination(gym.Wrapper):

    def __init__(self, name, max_length=1000):
        env = gym.make(name)
        self.max_length = max_length
        super().__init__(env)
    
    def reset(self):
        self._current_step = 0
        return super().reset()
        
    def step(self, act):
        obs, rew, done, info = super().step(act)
        self._current_step += 1
        done = self._current_step >= self.max_length
        return obs, rew, done, info