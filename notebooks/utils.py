from tqdm.notebook import tqdm
import random
import numpy as np

class ReplayMemory:
    def __init__(self, data, target):
        self.buffer = [data, target]
    
    def sample(self, batch_size):
        length = self.buffer[0].shape[0]
        batch = np.random.choice(np.arange(length), int(batch_size))
        return tuple(np.stack(data[batch]) for data in self.buffer)
    
    def batches(self, batch_size):
        length = self.buffer[0].shape[0]
        dataset = self.sample(length)
        i = 0
        while i < length:
            yield tuple(data[i:i+batch_size] for data in dataset)
            i += batch_size

class Args:
    data = {}
    def __init__(self, *args, **kwargs):
        self.data.update(dict(*args, **kwargs))
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return self.__dict__[key]

def rollout(steps, max_eps_length, env, agent):
    obs = env.reset()
    done = False
    eps_ret = 0
    t = 0
    
    obs_buf = []
    act_buf = []
    eps_ret_buf = []
    eps_len_buf = []
    
    for step in tqdm(range(steps)):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        eps_ret += reward
        cost = info.get("cost", 0)
        
        obs = next_obs
        

        if done or step == steps - 1 or t == max_eps_length - 1:
            eps_ret_buf.append(eps_ret)
            eps_len_buf.append(t + 1)
            
            obs = env.reset()
            done = False
            eps_ret = 0
            t = 0
        else:
            t += 1

        if step == steps - 1:
            break
            
    
    return dict(states=obs_buf, actions=act_buf, eps_ret=eps_ret_buf, eps_len=eps_len_buf)
