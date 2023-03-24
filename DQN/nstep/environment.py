'''
    Authored by : Dala Zhu
    Data : 2022.11.28
    Reference : https://github.com/higgsfield/RL-Adventure
'''

import math
import sys
import gym
import numpy as np
import pylab
import torch
from DDQN.Agent import Agent



EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 1000
epsilon_by_frame = lambda i: EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * i / EPSILON_DECAY)


class Environment:
    def __init__(self, env_id, render_mode):

        self.env = gym.make(env_id, render_mode = render_mode)

        self.num_action = self.env.action_space.n
        self.num_state = self.env.observation_space.shape[0]
        self.agent = Agent(self.num_state, self.num_action)

        self.n_frames = 10000000
        self.target_update_times = 500 # target network 更新步数
        self.reward_buffer = []

    def train(self):
        step_loss = []
        eposide_reward = 0
        eposide_count = 0 #第几幕
        # 初始化状态
        state, _ = self.env.reset()
        for i_frame in range(self.n_frames):
            #根据状态选行为
            epsilon = epsilon_by_frame(i_frame)

            action = self.agent.act(state, epsilon)

            next_state, reward, done, _, _ = self.env.step(action)

            self.agent.memo.push(state, action, reward, next_state, done)

            if done:
                state, _ = self.env.reset()
                self.reward_buffer.append(eposide_reward)
                eposide_count += 1
                # plot the play time
                pylab.figure(2)
                # pylab.plot(self.reward_buffer, 'b')
                # pylab.xlabel('eposide')
                # pylab.ylabel('eposide_reward')
                # pylab.savefig("./cartpole_dqn_reward.png")
                print("episode:", eposide_count, "  eposids_reward:", eposide_reward, "  epsilon:", epsilon)
                eposide_reward = 0
                if np.mean(self.reward_buffer[-min(10, len(self.reward_buffer)):]) > 400:
                    torch.save(self.agent.online_net, "./cartpole_dqn")
                    sys.exit()

            state = next_state
            eposide_reward += reward

            loss = self.agent.compute_td_loss()
            step_loss.append(loss.item())

            # pylab.figure(1)
            # pylab.plot(step_loss, 'r')
            # pylab.xlabel('step')
            # pylab.ylabel('loss')
            # pylab.savefig("./cartpole_dqn_loss.png")


            if i_frame % self.target_update_times == 0:
                self.agent.update_target()


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = Environment(env_id, 'human')
    env.train()
