import gym
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo
from matplotlib import animation
import time

# 训练过程存为gif文件
def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1)
    anim.save('./breakout_result.gif', writer='ffmpeg', fps=30)

def example():
    env = gym.make('MountainCar-v0',render = 'rgb_array')
    env.reset()
    env.render()
    env = RecordVideo(env,'videos',episode_trigger = lambda e : True)
    frames = []
    for episode in range(30):
        print(f'Starting episode {episode}')
        s = env.reset()
        frames.append(env.render(mode = 'rgb_array'))
        time.sleep(0.5)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
    env.close()
    display_frames_as_gif(frames)

if __name__ =="__main__":
    example()