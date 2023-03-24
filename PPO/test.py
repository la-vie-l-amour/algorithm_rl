import argparse
import os
import pylab
import torch
import gym
from IPython.core.display import clear_output
from gym.wrappers import RecordVideo
from model import ActorCritic
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=32,help = 'hidden size of model(actor and critic)')
    parser.add_argument('--env_name', type=str, default="Pendulum-v1")
    parser.add_argument('--continous_action_space', type= lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--continous_state_space', type= lambda x: x.lower() == 'true', default=True)

    args = parser.parse_args()
    return args

def save_fig(rewards, env_name):
    clear_output(True)  # 一定要有这行, 否则图片显示可能为空白
    fig = pylab.figure(figsize=(12, 10))
    pylab.plot(rewards, 'b')
    pylab.xlabel('step')
    pylab.ylabel('reward')
    graph_dir = f"./graphs/{env_name}"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    pylab.savefig(f"{graph_dir}/{env_name}_reward.jpg", bbox_inches="tight")
    pylab.close(fig)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env= gym.make(args.env_name, render_mode = 'rgb_array')
    env = RecordVideo(env, f'./videos/{args.env_name}',episode_trigger= lambda a: a==0, video_length=0)

    if args.continous_state_space:
        num_inputs = env.observation_space.shape[0]
    else:
        num_inputs = 1


    if args.continous_action_space:
        num_outputs = env.action_space.shape[0] #连续动作
    else:
        num_outputs = env.action_space.n #离散动作

    #加载模型
    model_path = f"./models/{args.env_name}/{args.env_name}.pt"
    print(model_path)
    model = ActorCritic(num_inputs, num_outputs, args.hidden_size, args.continous_action_space)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.to(device)
    model.eval()

    # 一些变量的初始化
    done = False
    total_reward = 0
    rewards = []

    state, _ = env.reset()
    while not done:
        if not args.continous_state_space:
            state = np.expand_dims(state, axis=-1)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().cpu().numpy()[0]

        next_state, reward, terminated,  truncated, info = env.step(action)
        state = next_state
        total_reward += reward
        rewards.append(reward)
        done = terminated or truncated  #一定要注意，done应是这样。
    env.close()
    save_fig(rewards, args.env_name)


if __name__ == '__main__':
    args = get_args()
    test(args)




