import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherSafetyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hazard_size = 0.05
        self.hazard_number = 3
        self.hazards = np.zeros(self.hazard_number * 2)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

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
    
    def generate_hazard(self, avoid):
        avoid = avoid.reshape(-1, 2)
        while True:
            valid = True
            hazard = self.np_random.uniform(low=-.2, high=.2, size=2)
            for item in avoid:
                if np.linalg.norm(item - hazard) < self.hazard_size * 2:
                    valid = False
                    break
            valid = valid and np.linalg.norm(hazard) < 0.2
            if valid:
                break
        return hazard
    
    def _get_cost(self):
        cost = 0
        pos = self.get_body_com("fingertip")[:2]
        hazards = self.hazards.reshape(-1, 2)
        for hazard in hazards:
            if np.linalg.norm(hazard - pos) < self.hazard_size:
                cost = 1
        return cost

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        
        self.hazards = np.array([])
        for _ in range(self.hazard_number):
            hazard = self.generate_hazard(np.concatenate([qpos, self.hazards]))
            self.hazards = np.append(self.hazards, hazard)
        
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        finger_pos = self.get_body_com("fingertip")[:2]
        hazard_obs = np.tile(finger_pos, self.hazard_number) - self.hazards
        
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
	    self.hazards,
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target"),
            hazard_obs,
        ])
