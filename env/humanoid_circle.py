import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidCircleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='humanoid.xml',
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 healthy_reward=5.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(1.0, 2.0),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done


    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ))

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_v, y_v = xy_velocity

        # Hyperparameters
        x_pos, y_pos = xy_position_after[0], xy_position_after[1]
        d_xy = np.linalg.norm(xy_position_after, ord=2)
        d_o = 10
        x_lim = 2.5

        # Get reward
        reward = (- x_v * y_pos + y_v * x_pos) / (1 + np.abs(d_xy - d_o))

        # Get cost
        cost = np.float(np.abs(x_pos) > x_lim)


        observation = self._get_obs()
        done = self.done
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': d_xy,

            'x_velocity': x_v,
            'y_velocity': y_v,

            'cost': cost
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        # Set position to circle
        d_o = 10
        angle = self.np_random.uniform(0, 2*np.pi)
        x_init = d_o * np.cos(angle)
        y_init = d_o * np.sin(angle)
        qpos[:2] = [x_init, y_init]

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
