import math
import random
import numpy as np
from hDQN.agent import Agent
import matplotlib.pyplot as plt

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 1000
epsilon_by_frame = lambda i: EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * i / EPSILON_DECAY)


# environment : Stochastic markov decission process
class StochasticMDP:
    def __init__(self):
        self.end = False  #whether visit 6 or not, i.e. if it visit the end state
        self.current_state = 2
        self.num_actions = 2
        self.num_states = 6
        self.p_right = 0.5
    def reset(self):
        self.end = False
        self.current_state = 2
        # exchange ndarray: one-hot
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, {}

    def step(self, action):
        reward = 0.0
        done = False # eposide
        # not terminal
        if self.current_state != 1:
            if action == 1: # right
                # right successfully ,s6 can not right move
                if random.random() < self.p_right and self.current_state < self.num_states:
                    self.current_state += 1
                # right unsuccessfully
                else:
                    self.current_state -= 1
            elif action == 0: # left
                self.current_state -= 1

            # if it visit the end state
            if self.current_state == self.num_states:
                self.end = True
        # terminal
        else:
            done = True
            if self.end:
                reward = 1.00
            else:
                reward = 1.00 / 100.00

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1

        return state, reward, done, {}, {}


class Environment:
    def __init__(self):

        self.env = StochasticMDP()

        self.num_action = self.env.num_actions
        self.num_state = self.env.num_states

        self.agent = Agent(self.num_state, self.num_action)

        self.n_frames = 10000000
        self.target_update_times = 1000 # target network 更新步数
        self.eposide_reward_buffer = []

    def to_onehot(self, x):
        vector_x = np.zeros(self.num_state)
        vector_x[x - 1] = 1
        return vector_x

    def train(self):

        eposide_reward = 0
        eposide_id = 0  # eposide-th
        # initial state
        state, _ = self.env.reset()
        done = False
        i_frame = 0
        while i_frame < self.n_frames:
            # choose action depends on state
            goal = self.agent.act(state, epsilon_by_frame(i_frame), self.agent.meta_online_net)

            # why is num_state
            onhot_goal = self.to_onehot(goal)

            meta_state = state
            extrinsic_reward = 0

            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onhot_goal])

                action = self.agent.act(goal_state, epsilon_by_frame(i_frame), self.agent.online_net)
                next_state, reward, done, _, _ = self.env.step(action)

                eposide_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

                self.agent.memo.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onhot_goal]))

                state = next_state

                self.agent.compute_td_loss(self.agent.online_net, self.agent.target_net,self.agent.optimizer, self.agent.memo)
                self.agent.compute_td_loss(self.agent.meta_online_net, self.agent.meta_target_net, self.agent.meta_optimizer,self.agent.meta_memo)

                i_frame += 1

                if i_frame % self.target_update_times == 0:
                    n = 100
                    plt.figure(figsize = (20, 5))
                    plt.title(i_frame)
                    plt.plot([np.mean(self.eposide_reward_buffer[i:i + n] for i in range(0, len(self.eposide_reward_buffer), n))])
                    plt.savefig("./reward.png")


            self.agent.meta_memo.push(meta_state, goal, extrinsic_reward, state, done)

            if done:
                state ,_  = self.env.reset()
                done = False
                self.eposide_reward_buffer.append(eposide_reward)
                eposide_reward = 0


if __name__ == '__main__':
    env = Environment()
    env.train()



