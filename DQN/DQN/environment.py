'''
    Reference: https://www.bilibili.com/video/BV1hF411L7qu/?spm_id_from=333.1007.top_right_bar_window_history.content.click
    Authored by: Dala Zhu
    Data: 2022.11.26
'''

import random
import sys

import numpy as np
import gym
import pylab
import torch
import torch.nn as nn
from DQN.agent import Agent

env = gym.make("CartPole-v1", render_mode = 'human')
s,_ = env.reset()  #s是ndarry类型

# hyperparameters
EPSILON_DECAY = 100000 #衰减是按步来衰减
EPSILON_START = 1.0
EPSILON_END = 0.02

TARGET_UPDATE_FREQURNCY  = 10 # target network updata frequency

n_episode = 5000 # 5000局游戏，或者是5000幕
n_time_step = 1000 # 每局1000步

n_state = len(s)  # 表示状态（ndarry）的维度，这里是4
n_action = env.action_space.n # action的维度，action是Discrete(2)
agent = Agent(n_input = n_state, n_output= n_action) #初始化agent

REWARD_BUFFER = np.empty(shape = n_episode)
# losses = []
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        # epsilon greed ，explore，exploit,探索率，用差分
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()  # [0,1)中的随机数
        if random_sample <= epsilon:
            a = env.action_space.sample() # explore
        else:
            a = agent.online_net.act(s) # maxQ

        s_, r, done, _, info = env.step(a)  #done是是否这局游戏结束

        agent.memo.add_memo(s, a, r, done, s_) #将transitions加入replay
        s = s_    #状态转移
        episode_reward += r #累积奖励


        # 判断游戏是否结束
        if done:
            s,_ = env.reset() #环境重置
            REWARD_BUFFER[episode_i] = episode_reward #记录这一幕的奖励
            # pylab.figure(1)
            # pylab.plot(REWARD_BUFFER, 'b')
            # pylab.xlabel('eposide')
            # pylab.ylabel('eposide_reward')
            # pylab.savefig("./cartpole_dqn_reward.png")
            print("episode:", episode_i, "  eposids_reward:", episode_reward, "  epsilon:", epsilon)

            if np.mean(REWARD_BUFFER[:episode_i]) >= 500:
                sys.exit()
            break # 进入下一个幕


        # 这一段代码怎么理解，应该当经验>=200说明，已经训练好了，之后无需在进行训练
        # if np.mean(REWARD_BUFFER[:episode_i]) >= 200:
        #     while True:
        #         a = agent.online_net.act(s)
        #         s, r, done, _, info = env.step(a)
        #         env.render()
        #         if done:
        #             s,_ = env.reset()


        # minibatch
        batch_s, batch_a, batch_r, batch_done, batch_s_= agent.memo.sample() #采用

        #compute target,y_i
        target_q_values = agent.target_net(batch_s_)

        max_target_q_values = target_q_values.max(dim = 1, keepdim = True)[0] #[0]是得到values，丢弃indices
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

        # compute q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input = q_values, dim = 1, index = batch_a)


        # compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)
        # losses.append(loss.item())

        # pylab.figure(2)
        # pylab.plot(losses, 'r')
        # pylab.xlabel('eposide')
        # pylab.ylabel('eposide_loss')
        # pylab.savefig("./cartpole_dqn_loss.png")

        # Gradient descent
        agent.optimizer.zero_grad() #将模型的参数初始化为0， 为何？？？
        loss.backward() # back propagation
        agent.optimizer.step() # updata parameters


    if episode_i % TARGET_UPDATE_FREQURNCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())










