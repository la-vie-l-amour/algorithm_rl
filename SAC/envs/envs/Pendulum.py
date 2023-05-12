import gym
import numpy as np


class Pendulum:

    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self, render_mode = 'rgb_array'):

        self.env = gym.make("Pendulum-v1",render_mode = render_mode)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, term, trun, info = self.env.step(action)
        done = np.bitwise_or(term, trun)
        return state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def sample_action(self):
        return self.env.action_space.sample()

