'''
    Authored by : Dala Zhu
    Data : 2022.11.28
    Reference : https://github.com/higgsfield/RL-Adventure
'''
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque  #队列，先进先出,如果超过maxlen，会自动删除旧的，


# experience replay
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        len_sample = min(len(self.buffer), batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*random.sample(self.buffer, len_sample)) #random.sample返回列表
        # array转为tensor
        batch_s_tensor = torch.as_tensor(np.asarray(batch_state), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)  # unsqueeze 进行升维
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_next_state), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor, batch_done_tensor

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_state, num_action):
        super(DQN, self).__init__()

        self.num_state = num_state
        self.num_action = num_action

        self.layers = nn.Sequential(
            nn.Linear(self.num_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_action)
        )

    def forward(self, x):
        return self.layers(x)




class Agent:
    def __init__(self, num_state, num_action):
        self.device = torch.device('cuda')

        self.num_action = num_action
        self.gamma = 0.99
        self.batch_size = 32

        self.online_net = DQN(num_state= num_state,num_action=num_action).to(self.device)

        self.target_net = DQN(num_state= num_state, num_action=num_action).to(self.device)

        self.memo = ReplayBuffer(1000)
        self.learning_rate = 1e-3

        #RMSprop
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr= self.learning_rate)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)# [1,2,3,4]->[[1,2,3,4]]
            state = state.to(self.device)
            q_value = self.online_net(state)  # [[1,2]]这种形式
            max_q_idx = torch.argmax(input=q_value)
            action = max_q_idx.detach().item()
        else:
            action = random.randrange(self.num_action)
        return action

    def compute_td_loss(self):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.memo.sample(self.batch_size)

        # compute TD target ,common ,y_i

        # next_q_values = self.agent.target_net(batch_next_state)
        # next_q_value = next_q_values.max(1, keepdim=True)[0]
        # expected_q_value = batch_reward + self.gamma * next_q_value * (1 - batch_done)

        #cuda
        batch_next_state = batch_next_state.to(self.device)
        batch_state = batch_state.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_done = batch_done.to(self.device)
        batch_action = batch_action.to(self.device)

        # compute double DQN , y_i
        next_q_values_online = self.online_net(batch_next_state)
        next_q_values_target = self.target_net(batch_next_state)

        next_q_value = next_q_values_target.gather(dim=1, index=torch.max(next_q_values_online, dim=1, keepdim=True)[1])

        expected_q_value = batch_reward + self.gamma * next_q_value * (1 - batch_done)

        # compute Q(s,a)

        q_values = self.online_net(batch_state)
        q_value = q_values.gather(1, batch_action)


        # compute loss
        loss = (q_value - expected_q_value.data).pow(2).mean()
        loss = loss.to(self.device)

        # loss = nn.functional.smooth_l1_loss(expected_q_value, q_value)

        # gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())





