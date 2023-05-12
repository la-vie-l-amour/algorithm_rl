import gym
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, continous_action_space , std=0.0):
        super(ActorCritic, self).__init__()

        self.continous_action_space = continous_action_space

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        if self.continous_action_space:
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        if self.continous_action_space:
            mu = self.actor(x)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            action = self.actor(x)
            probs = F.softmax(action, dim=-1)
            dist = Categorical(probs)
        return dist, value
