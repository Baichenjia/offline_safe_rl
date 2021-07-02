import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherSafetyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.contact_dist = 0.06 + 0.01
        utils.EzPickle.__init__(self)
        path = os.path.join(os.path.dirname(__file__), "assets", "reacher_safety.xml")
        mujoco_env.MujocoEnv.__init__(self, path, 2)
        self.hazard = np.array([.1, .1])

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        cost = self._get_cost()
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, cost=cost)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
    
    def _get_cost(self):
        cost = 0
        vec = self.get_body_com("fingertip") - self.get_body_com("hazard")
        if np.linalg.norm(vec) < self.contact_dist:
            cost = 1
        return cost

    def reset_model(self):
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if (np.linalg.norm(self.goal) < 0.2 and
                np.linalg.norm(self.hazard - self.goal) > self.contact_dist):
                break
        while True:
            self.arm = self.np_random.uniform(low=-.2, high=.2, size=2)
            if (np.linalg.norm(self.goal) < 0.2 and
                np.linalg.norm(self.hazard - self.goal) > self.contact_dist):
                break
        qpos = np.concatenate([self.arm, self.goal, self.hazard])

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target"),
            self.get_body_com("fingertip") - self.get_body_com("hazard"),
        ])
