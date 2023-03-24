import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# how share parameters in two networks (programming)?
class ValueNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, init_w = 3e-3):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    def forward(self, state, action):
        x = torch.cat((state, action), dim = 1)
        x = self.layers(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.layers(state)
        x = self.linear3(x)
        x = self.tanh(x)
        return x

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
        batch_a_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.int64)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)  # unsqueeze 进行升维
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_next_state), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor, batch_done_tensor

    def __len__(self):
        return len(self.buffer)




class Agent:
    def __init__(self, num_states, nums_actions, hidden_dim):
        self.num_action = nums_actions
        self.device = torch.device('cuda')

        self.value_net = ValueNetwork(num_states = num_states, num_actions = nums_actions, hidden_size = hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(num_states = num_states, num_actions = nums_actions, hidden_size = hidden_dim).to(self.device)

        self.target_value_net = ValueNetwork(num_states = num_states, num_actions = nums_actions, hidden_size = hidden_dim).to(self.device)
        self.target_policy_net = PolicyNetwork(num_states = num_states, num_actions = nums_actions, hidden_size = hidden_dim).to(self.device)

        # parameter initial
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.value_lr = 1e-3
        self.policy_lr = 1e-4

        self.value_optimizer = optim.Adam(params = self.value_net.parameters(), lr = self.value_lr)
        self.policy_optimizer = optim.Adam(params = self.policy_net.parameters(), lr = self.policy_lr)

        self.replay_buffer_size = 10000
        self.replay_buffer = ReplayBuffer(capacity = self.replay_buffer_size)

        self.value_criterion = nn.MSELoss()

        self.gamma = 0.99
        self.batch_size = 128


    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)# [1,2,3,4]->[[1,2,3,4]]
        state = state.to(self.device)
        action = self.policy_net(state)  # [[1]]这种形式
        return action[0].detach().cpu().numpy()

    def compute_loss(self,min_value = -np.inf, max_value = np.inf, soft_tau = 1e-2):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(self.batch_size)
        #cuda
        next_state = batch_next_state.to(self.device)
        state = batch_state.to(self.device)
        reward = batch_reward.to(self.device)
        done = batch_done.to(self.device)
        action = batch_action.to(self.device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state,next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)
        # print(state.shape, action.shape)
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        # update target net parameters
        self.update_target_net(soft_tau)

    # update target network parameters
    def update_target_net(self,soft_tau):

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
