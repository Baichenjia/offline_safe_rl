import gym
import pygame
from gym import spaces
from gym.utils import seeding
import numpy as np

class PointMass(gym.Env):
    def __init__(self, N=1):
        # Step 1: Car parameterss
        self.v_max = 0.1
        self.v_sigma = 0.01
        # Step 3: Environment parameters
        self.d_safe = 0.1
        self.d_goal = 0.05
        self.d_sampling = 0.1

        self.init_pos = np.array([1.0, 1.0])

        self.N = N # number of obstacles

        self.low_state = 0
        self.high_state= 1

        self.min_actions = np.array(
            [-self.v_max, -self.v_max], dtype=np.float32
        )
        self.max_actions = np.array(
            [self.v_max, self.v_max], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2+2*N+1+2, ),
            dtype=np.float32
        )

        self.goal = np.array([0.1, 0.1])
        self.r = 0.3 # obstacle radius
        self.centers = None

        # Step 4: Rendering parameters
        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [255, 255, 255]
        self.wall_color = [0, 0, 0]
        self.circle_color = [255, 0, 0]
        self.safe_circle_color = [200,0,0]
        self.lidar_color = [0, 0, 255]
        self.goal_color = [0, 255, 0]
        self.robot_color = [0, 0, 0]
        self.safety_color = [255, 0, 0]
        self.goal_size = 15
        self.radius = 9
        self.width = 3
        self.pygame_init = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self):
        if self.centers is None:
            self.generate_obstacles(r=self.r)

        sampled = False
        while not sampled:
            self.init_pos = self.np_random.uniform(0.1, 0.9, size=(2,))
            if self.is_safe(self.init_pos):
                sampled = True
        # self.init_pos = np.array([0.9, 0.9]) + self.np_random.uniform(-0.1, 0, size=(2,))

        self.state = np.array(list(self.init_pos) + list(self.centers.flatten()) + [self.r] + list(self.goal))
        return np.array(self.state)

    def generate_obstacles(self, r=0.3):
        centers = self.np_random.uniform(low=self.low_state+self.d_sampling, high=self.high_state-self.d_sampling, size=(self.N, 2))
        centers[self.N-1][0] = float(self.high_state / 2)
        centers[self.N-1][1] = float(self.high_state / 2)
        self.centers = centers
        self.circles = [np.array([center[0], center[1], self.r]) for center in centers]

    def get_dist_to_goal(self, state):
        return np.linalg.norm(state[-2:]-state[:2])

    # Check if the state is safe.
    def is_safe(self, state):
        if len(state.shape) == 1:
            safe = True
            for (p1,p2,r) in self.circles:
                d_circle = (state[0]-p1)**2 + (state[1]-p2)**2
                if d_circle <= (r ** 2):
                    safe = False
            return safe
        elif len(state.shape) == 2:
            # hack since there is only one circle
            p1, p2, r = self.circles[0]
            d_circle = (state[:, 0] - p1) ** 2 + (state[:, 1] - p2) ** 2
            safe = (d_circle > r ** 2).astype(float)
            return safe

    def step(self, action):

        action = np.clip(action, -self.v_max, self.v_max)
        assert self.action_space.contains(action)
        # action += np.random.normal(0.0, self.v_sigma)

        d_goal = self.get_dist_to_goal(self.state)
        reward = - d_goal - 0.1
        done = 0
        if d_goal < self.d_goal:
            done = 1
            # reward += 10

        self.state[:2] = self.state[:2] + action
        # self.state[2:4] = self.state[2:4] + self.action_space.sample() / 10
        # self.circles[0][:2] = self.state[2:4]
        self.state = np.clip(self.state, self.low_state, self.high_state)

        return np.array(self.state), reward, done, {}

    def render(self, mode=None):
        # Step 1: Initialize pygame if necessary
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        # Step 2: Get events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Step 3: Render screen
        self.screen.fill(self.background_color)

        # Step 4: Extract state
        p_car = self.state[:2]

        # Step 9: Render robot
        p = (self.screen_scale * p_car).astype(int).tolist()
        pygame.draw.circle(self.screen, self.robot_color, p, self.radius, self.width)

        for circle in self.circles:
            c, r = (self.screen_scale*circle[:2]).astype(int), int(self.screen_scale*circle[2])
            pygame.draw.circle(self.screen, self.circle_color, c, r)
            # pygame.draw.circle(self.screen, self.safe_circle_color, c, self.screen_scale * r+int(self.d_safe*self.screen_scale), int(self.d_safe*self.screen_scale))

        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), self.goal_size)
        # Step 11: Render pygame
        pygame.display.flip()

        # Step 12: Pause
        self.clock.tick(20)


if __name__ == '__main__':
    import gym
    env = PointMass()
    env.seed(0)
    for i in range(100):
        state = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            state, _, _, _ = env.step(action)
            env.render()