'''

运行 python train.py --env_name="FrozenLake-v1" --continous_action_space=False --continous_state_space=False --threshold_reward=0.7 --mini_batch=128
python train.py --env_name="Taxi-v3" --continous_action_space=False --continous_state_space=False --threshold_reward=-201 这个不对，要改进


'''

import argparse
from PPO.env import SubprocVecEnv
from model import ActorCritic
import numpy as np
import torch
import pylab
import os
import torch.optim as optim
from IPython.display import clear_output
import gymnasium as gym

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=32,help = 'hidden size of model(actor and critic)')
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-2) #试着调大这个超参，或是5e-2,效果都不错
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--ppo_epochs', type=int, default=30)
    parser.add_argument("--threshold_reward", type=float, default=-200)
    parser.add_argument("--num_steps", type=int, default=128,help = 'used to generate pairs')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--clip_param', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument("--max_frames", type=int, default=15000)
    parser.add_argument('--beta', type=float, default=0.001, help='entropy coefficient')
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--continous_action_space", type=lambda x: x.lower() == 'true', default=True)  #这个类型必须如此，不能写bool，否则会报错
    parser.add_argument("--continous_state_space", type = lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()
    return args


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk


def path_isExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot(frame_idx, rewards, env_name):
    clear_output(True)
    fig = pylab.figure(figsize =(12,10))
    pylab.plot(rewards, 'b')
    pylab.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    path_isExist(f"./graphs/{env_name}")
    pylab.savefig(f"./graphs/{env_name}/{env_name}.png", bbox_inches = "tight")
    pylab.close(fig)


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval(model, env_name, continous_state_space, vis=False):
    env = gym.make(env_name)
    with torch.no_grad():

        state, _ = env.reset()
        if vis:
            env.render()
        done = False
        total_reward = 0
        while not done:
            if not continous_state_space:
                state = np.expand_dims(state, axis=-1)

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            next_state, reward, term, truncated, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis:
                env.render()
            total_reward += reward
            done = term or truncated
    env.close()
    return total_reward


def main(args):

    frame_idx = 0
    test_rewards = []
    envs = [make_env(args.env_name) for _ in range(args.num_envs)]
    envs = SubprocVecEnv(envs)

    state= envs.reset()

    early_stop = False

    if args.continous_state_space:
        num_inputs = envs.observation_space.shape[0]
    else:
        num_inputs = 1

    if args.continous_action_space:
        num_outputs = envs.action_space.shape[0]
    else:
        num_outputs = envs.action_space.n

    model = ActorCritic(num_inputs, num_outputs, args.hidden_size, args.continous_action_space).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    optimizer = optim.Adam([
        {'params': model.actor.parameters(), 'lr': args.lr_actor},
        {'params': model.critic.parameters(), 'lr': args.lr_critic}
    ])

    model.train() #这一行代码是否需要

    while frame_idx < args.max_frames and not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0
        # global next_state
        global next_state
        # collect data
        for _ in range(args.num_steps):

            state = torch.FloatTensor(state).to(device)

            if not args.continous_state_space:
                state = state.unsqueeze(dim=-1)

            dist, value = model(state)
            action = dist.sample()

            next_state, reward, term, trun, _  = envs.step(action.cpu().numpy())

            # done = np.bitwise_or(term , trun)

            done = term #如果done是上面的那个，则训练过程会很慢，或者说训练效果很不好,这个训练是对于Pendulum-v1而言的

            log_prob = dist.log_prob(action)

            if not args.continous_action_space:
                log_prob = log_prob.unsqueeze(dim = -1)

            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)

            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)

            if not args.continous_action_space:
                action = action.unsqueeze(dim = -1)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if frame_idx % 1000 == 0:
                test_reward = np.mean([eval(model, args.env_name, args.continous_state_space) for _ in range(10)])  #这里是否可以传参数model
                test_rewards.append(test_reward)
                print('++++',test_reward,'+++++',frame_idx,"++++")
                plot(frame_idx, test_rewards, args.env_name)

                if test_reward > args.threshold_reward:
                    early_stop = True

                    #保存模型
                    path_isExist(f"./models/{args.env_name}")
                    torch.save(model.state_dict(), f"./models/{args.env_name}/{args.env_name}.pt")

        if not args.continous_state_space:
            next_state = np.expand_dims(next_state, axis=-1)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values, args.gamma, args.tau)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values

        # PPO update
        for _ in range(args.ppo_epochs):
            for state_, action_, old_log_probs, return_, advantage in ppo_iter(args.mini_batch_size, states, actions,
                                                                             log_probs, returns, advantages):
                dist, value = model(state_)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action_)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()  # critic_loss 关于这个

                loss = 0.5 * critic_loss + actor_loss - args.beta * entropy #这里将entropy放入loss，是为了增大它的熵，也即扩大explore
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# 产生mini_batch_size数据用于更新参数
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]


# Generalized Advantage Estimator 一种用于估计Advantage的方法
def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


if __name__ =='__main__':
    args = get_args()
    main(args)
