'''
    Reference : https://github.com/rlcode/per
    Date: 2022.11.30
    Authored: Dala zhu
'''


import random
import math
import sys
import gymnasium as gym
import numpy as np
import numpy
import pylab
import torch
import torch.nn as nn

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 1000
epsilon_by_frame = lambda i : EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * i / EPSILON_DECAY)

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)  #start from 0
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0  #tree的leaf中被占用的个数，也即data中不为空的个数

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)


    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)


    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



class Memory:  # stored as ( s, a, r, s_, done) in SumTree
    e = 0.01  #ebusilon
    a = 0.6 #alpha
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):   #sample is (s, a, r, s_, done)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []

        segment = self.tree.total() / n  #抽取时为何会进行这样化等分抽取？
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  #每次采样，beta都会增加，为何？

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)   # sample from [a,b]
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, - self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight #idxs is the index in tree.tree, not in tree.data


    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class DuelingDQN(nn.Module):
    def __init__(self, n_state, n_action):

        super(DuelingDQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action

        self.feature = nn.Sequential(
            nn.Linear(self.n_state, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_action)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()  # V(s) + A(s,a) - mean(A(s,a)),这里是将max(A(s,a))换成了mean(A(s,a))



class Agent:
    def __init__(self, num_state, num_action):

        self.num_state = num_state
        self.num_action = num_action
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.batch_size = 64  # 如果太大训练不好训练
        self.load_model = False #超参，是否记载模型

        self.online_net = DuelingDQN(num_state, num_action)

        if self.load_model:
            self.online_net = torch.load('./save_model/cartpole_dqn')

        self.target_net = DuelingDQN(num_state, num_action)
        self.update_target()

        self.memo = Memory(capacity = 1000)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr= self.learning_rate)


    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.as_tensor(state).unsqueeze(0)
            q_value = self.online_net(state)
            # action = q_value.max(1)[1].item()
            max_q_idx = torch.argmax(input=q_value)
            action = max_q_idx.detach().item()
        else:
            action = random.randrange(self.num_action)
        return action



    def push(self, state, action, reward, next_state, done):
        # error最大，这个需要看论文确定
        error = max(self.memo.tree.tree[self.memo.capacity - 1: self.memo.tree.n_entries + self.memo.tree.capacity])
        self.memo.add(error = error, sample = (state, action, reward, next_state, done))


    def compute_td_loss(self):

        batch, idxs, is_weight = self.memo.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        batch_state_tensor = torch.as_tensor(np.asarray(batch_state), dtype=torch.float32)
        batch_action_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.int64).unsqueeze(-1)
        batch_reward_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)  # unsqueeze 进行升维
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_next_state_tensor = torch.as_tensor(np.asarray(batch_next_state), dtype=torch.float32)

        # compute double DQN , y_i
        next_q_values_online = self.online_net(batch_next_state_tensor)
        next_q_values_target = self.target_net(batch_next_state_tensor)

        next_q_value = next_q_values_target.gather(dim=1, index=torch.max(next_q_values_online, dim=1, keepdim=True)[1])

        pred_q_value = batch_reward_tensor + self.gamma * next_q_value * (1 - batch_done_tensor)

        # compute Q(s,a)
        q_values = self.online_net(batch_state_tensor)
        true_q_value = q_values.gather(1, batch_action_tensor)

        # update priority
        errors = torch.abs(pred_q_value - true_q_value).squeeze(dim = 1).detach().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memo.update(idx, errors[i])

        # compute loss
        loss = (true_q_value - pred_q_value.data).pow(2).mean()
        # loss = nn.functional.smooth_l1_loss(expected_q_value, q_value)

        # gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    # update target net parameters
    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())



class Environment:
    def __init__(self, env_id, render_mode):
        self.env = gym.make(env_id, render_mode = render_mode)

        self.num_action = self.env.action_space.n
        self.num_state = self.env.observation_space.shape[0]
        self.agent = Agent(self.num_state, self.num_action)
        self.updata_target_frame = 500
        self.n_episode = 500


    def train(self):
        eposide_rewards = []
        eposide_ids = []
        frame_losses = []
        frames = 0 # 记录所有的帧数
        for i_eposide in range(self.n_episode):
            done = False
            eposide_reward = 0
            # 初始化状态
            state, _ = self.env.reset()
            while not done:
                frames += 1
                epsilon = epsilon_by_frame(frames)
                action = self.agent.act(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)

                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or eposide_reward == 499 else -10

                # add transitions to memory
                self.agent.push(state, action, reward, next_state, done)

                #compute td loss
                loss = self.agent.compute_td_loss()
                frame_losses.append(loss.item())

                # pylab.figure(1)
                # pylab.plot(frame_losses, 'r')
                # pylab.xlabel('step')
                # pylab.ylabel('loss')
                # pylab.savefig("./save_graph/cartpole_dqn_loss.png")

                eposide_reward += reward
                state = next_state

                if done:
                    eposide_reward = eposide_reward if eposide_reward == 500 else eposide_reward + 10
                    eposide_rewards.append(eposide_reward)
                    eposide_ids.append(i_eposide)

                    # plot the play time
                    # pylab.figure(2)
                    # pylab.plot(eposide_ids, eposide_rewards, 'b')
                    # pylab.xlabel('eposide')
                    # pylab.ylabel('eposide_reward')
                    # pylab.savefig("./save_graph/cartpole_dqn_reward.png")

                    print("episode:", i_eposide, "  eposids_reward:", eposide_reward, "  memory length:",
                          self.agent.memo.tree.n_entries, "  epsilon:", epsilon)

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(eposide_rewards[-min(10, len(eposide_rewards)):]) > 490:
                        torch.save(self.agent.online_net, "./save_model/cartpole_dqn")
                        sys.exit()

                if frames % self.updata_target_frame == 0:
                    self.agent.update_target()


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = Environment(env_id, 'human')
    env.train()
