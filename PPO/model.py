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

'''

# 重新调整一下代码

class PPO:
    def __init__(self, num_state, num_action, hidden_size, lr,lr_actor, lr_critic, mini_batch_size, ppo_epochs, clip_param, beta ,gamma, tau):
        self.ActorCritic = ActorCritic(num_state, num_action, hidden_size)
        self.optimizer = torch.optim.Adam(self.ActorCritic.parameters(), lr= lr)

        # 给不同的网络设置不同的学习率
        self.optimizer = torch.optim.Adam([
            {'params': self.ActorCritic.actor.parameters(), 'lr': lr_actor},
            {'params': self.ActorCritic.critic.parameters(), 'lr': lr_critic}
        ])

        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, state):
        dist, value = self.ActorCritic(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


    def eval(self, env_name, vis):
        env = gym.make(env_name)
        state, _ = env.reset()
        if vis:
            env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.ActorCritic(state)
            next_state, reward, term, truncated, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis:
                env.render()
            total_reward += reward
            done = term or truncated
        env.close()
        return total_reward


    #选择mini_batch
    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        ids = np.random.permutation(batch_size)
        ids = np.split(ids, batch_size // self.mini_batch_size)
        for i in range(len(ids)):
            yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

    def ppo_update(self):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(self.mini_batch_size, states, actions,
                                                                             log_probs, returns, advantages):
                dist, value = self.ActorCritic(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()  # critic_loss 关于这个

                loss = 0.5 * critic_loss + actor_loss - self.beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



    def compute_gae(self): #next_value, rewards, masks, values, gamma, tau
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
        
'''