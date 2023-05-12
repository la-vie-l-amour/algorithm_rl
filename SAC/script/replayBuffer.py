import random
import gym
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # state, action, reward, next_state, done = zip(*self.buffer)
        # state = np.array(state)
        # action = np.array(action)
        # reward = np.array(reward)
        # next_state = np.array(next_state)
        # done = np.array(done)
        # L = len(self.buffer)
        # ids = np.random.permutation(L)
        # for i in range(0, L, batch_size):
        #     j = min(L, i + batch_size)
        #     batch_ids = ids[i:j]
        #     yield state[batch_ids, :], action[batch_ids, :], reward[batch_ids], next_state[batch_ids, :], done[batch_ids]

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)




class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

