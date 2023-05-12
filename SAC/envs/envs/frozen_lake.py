import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

# 动作设为(4,)的numpy，然后进行softmax和argmax
class FrozenLake:
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self,render_mode = "rgb_array"):
        self.env = gym.make("FrozenLake-v1",render_mode = render_mode)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n


    def reset(self):
        state,_ = self.env.reset()
        return self.encode_vector(state, self.n_states)

    def step(self, action):
        #这个softmax是否需要进行取绝对值，然后再进行，可以运行一下 再进行考虑
        action = self.decode_vector(action)

        state, reward, term, trun, info= self.env.step(action)
        done = np.bitwise_or(term, trun)

        state = self.encode_vector(state, self.n_states)
        return state, reward, done, info

    def close(self):
        self.env.close()
    def render(self):
        self.render()

    @property
    def action_space(self):
        return np.zeros(self.n_actions)
        # return gym.spaces.Box(low = 0, high = 2,shape=(4,), dtype= np.int32)

    @property
    def observation_space(self):
        return np.zeros(self.n_states)
        # return gym.spaces.Box(low = 0, high = 15, shape = (1,),dtype = np.int32)

    def sample_action(self):
        indx = self.env.action_space.sample()
        return self.encode_vector(indx, self.n_actions)

    def encode_vector(self, index, dim):
        vector_encoded = np.random.randn(dim)  #用标准正态分布生成
        vector_encoded[index] = 1
        return vector_encoded
    def decode_vector(self,vector_decoded):
        vector_decoded_softmax = F.softmax(torch.tensor(vector_decoded), dim=0)
        index = torch.argmax(vector_decoded_softmax, dim=0).item()
        return index







