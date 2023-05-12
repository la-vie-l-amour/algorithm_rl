import pickle

from envs import make_env,SubprocVecEnv, get_config,Env

import argparse

import torch
import numpy as np

from torch.optim import Adam
import os
from replayBuffer import ReplayBuffer, NormalizedActions
from IPython.display import clear_output
import pylab
import torch.nn.functional as F
import gym
from model import QNetwork, GaussianPolicy, DeterministicPolicy
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def path_isExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def plot(frame_idx, rewards,stds, env_name):
    clear_output(True)
    fig = pylab.figure(figsize =(12,10))
    pylab.plot(rewards, 'b')
    pylab.xlabel('iteration')
    pylab.ylabel('reward')
    pylab.title(f'SAC {np.mean(np.array(rewards))}')
    path_isExist(f"./graphs/{env_name}")
    pylab.savefig(f"./graphs/{env_name}/{env_name}.png", bbox_inches = "tight")
    pylab.close(fig)
    # 保存rewards
    with open("./graphs/rewads_buffer", 'wb') as f:
        pickle.dump(rewards, f)
    with open("./graphs/stds_buffer", 'wb') as f:
        pickle.dump(stds, f)

def eval(model, env_name, max_episode_len, vis=False):
    env = Env(env_name, max_episode_len)
    with torch.no_grad():
        state = env.reset()
        if vis:
            env.render()
        done = False
        total_reward = 0
        while not done:
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if vis:
                env.render()
            total_reward += reward
    env.close()
    return total_reward

def main(args):
    envs = Env(args.env_name,args.max_episode_len)

    num_actions = envs.action_space.shape[0]
    num_states = envs.observation_space.shape[0]

    critic = QNetwork(num_states, num_actions, args.hidden_size).to(device=device)
    critic_optim = Adam(critic.parameters(), lr=args.lr)

    critic_target = QNetwork(num_states, num_actions, args.hidden_size).to(device)
    hard_update(critic_target, critic)

    if args.policy_type == "Gaussian":
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if args.automatic_entropy_tuning is True:
            target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = Adam([log_alpha], lr=args.lr)

        policy = GaussianPolicy(num_states, num_actions, args.hidden_size, envs.action_space).to(device)
        policy_optim = Adam(policy.parameters(), lr=args.lr)

    else:
        args.alpha = 0
        args.automatic_entropy_tuning = False

        policy = DeterministicPolicy(num_states, num_actions, args.hidden_size, envs.action_space).to(device)
        policy_optim = Adam(policy.parameters(), lr=args.lr)


    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    frame_idx = 0
    rewards = []
    stds = []

    while frame_idx < args.max_frames :
        state_ = envs.reset()
        for _ in range(args.num_steps):
            action_ = policy.select_action(state_)
            next_state_, reward_, done_, _ = envs.step(action_)

            replay_buffer.push(state_, action_, reward_, next_state_, done_)

            state_ = next_state_
            frame_idx += 1

        for _ in range(args.train_epochs):
            # for state, action, reward, next_state, done in replay_buffer.sample(args.batch_size):
            if len(replay_buffer) > args.batch_size:   #改
                state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)

                state_batch = torch.FloatTensor(state).to(device)
                next_state_batch = torch.FloatTensor(next_state).to(device)
                action_batch = torch.FloatTensor(action).to(device)
                reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(device)
                done_batch = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = policy.sample(next_state_batch)
                    qf1_next_target, qf2_next_target = critic_target(next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - args.alpha * next_state_log_pi
                    next_q_value = reward_batch + (1 - done_batch) * args.gamma * (min_qf_next_target)
                qf1, qf2 = critic(state_batch,
                                       action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1_loss = F.mse_loss(qf1,
                                      next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2,
                                      next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf_loss = qf1_loss + qf2_loss

                critic_optim.zero_grad()
                qf_loss.backward()
                critic_optim.step()

                pi, log_pi, _ = policy.sample(state_batch)

                qf1_pi, qf2_pi = critic(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss = ((args.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                policy_optim.zero_grad()
                policy_loss.backward()
                policy_optim.step()

                if args.automatic_entropy_tuning:
                    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                    args.alpha = log_alpha.exp()

                soft_update(critic_target, critic, args.soft_tau)
        if frame_idx % 100 ==0:
            reward_list = [eval(policy, args.env_name, args.max_episode_len) for _ in range(10)]
            episode_reward = np.mean(reward_list)
            episode_std = np.std(reward_list)
            rewards.append(episode_reward)
            stds.append(episode_std)
            print(episode_reward, " +++++ ", frame_idx)
            plot(frame_idx, rewards,stds, args.env_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str,
                        default="Vehicle")  # Pendulum-v1, SparseMountainCar-v0, FrozenLake-v1，Vehicle
    args = parser.parse_args()  # args, unknown = parser.parse_know_args()
    config = get_config(args)
    main(config)