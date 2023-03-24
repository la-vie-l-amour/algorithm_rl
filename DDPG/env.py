# Normalize action space
import gym
import numpy as np
from DDPG.agent import Agent
import matplotlib.pyplot as plt


# Ornstein-Uhlenbeck process， one of the stochastic process

class OUNoise(object):
    def __init__(self, action_space, mu = 0.0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):
        self.mu = mu
        self.theata = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theata * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


#这里的normalizedActions完全不理解，呃呃呃。为何要这样计算，而且这里为何要对行动进行标准化，而不是对状态进行标准化。
class NormalizedAcions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def _reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


class Environment:
    def __init__(self):
        self.env = NormalizedAcions(gym.make("Pendulum-v1", render_mode= 'human'))
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.hidden_size = 256
        self.agent = Agent(self.num_states,self.num_actions,self.hidden_size)
        self.ou_noise = OUNoise(self.env.action_space)

    def plot(self,id_frame, eposide_rewards):
        plt.figure(figsize=(20, 5))
        plt.title('frame %s. reward: %s' % (id_frame, eposide_rewards[-1]))
        plt.plot(eposide_rewards)
        plt.show()

    def train(self, max_frames = 12000, max_steps = 500):
        eposide_rewards = []

        id_frame = 0
        while id_frame < max_frames:
            state, _ = self.env.reset()
            self.ou_noise.reset()
            eposide_reward = 0

            for step in range(max_steps):
                action = self.agent.get_action(state) #action is ndarry
                action = self.ou_noise.get_action(action, step)

                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.replay_buffer.push(state,action,reward,next_state,done)
                self.agent.compute_loss()
                state = next_state
                eposide_reward += reward
                id_frame += 1

                if id_frame % max(1000, max_steps + 1) == 0:
                    self.plot(id_frame, eposide_rewards)
                if done:
                    break
            eposide_rewards.append(eposide_reward)

if __name__ == '__main__':
    env = Environment()
    env.train()