'''
任务卸载环境
动作空间、状态空间和reward functions
动作空间：5个基站，每个基站的通信和计算力不同，所以动作空间为5，离散动作


状态空间： （5x2），每个基站都有信噪比和计算能力，模型根据用户的delay和每个基站的状态，选择将任务卸载到哪个BS中,,    //状态的shape应该是5x2
[f1,f2,f3,f4,f5,v1,v2,v3,v4,v5] ,v的可能取值[1,3]，f的可能取值为[4,6,8,10,12]


Delay: 用户的delay可以设置为随机产生或者按照泊松分布产生，公式参考论文，目的是为了保证用户delay，然后进行任务卸载

'''

import numpy as np
import random
import torch
import torch.nn.functional as F

from gym.spaces import Box

#计算reward 的超参
tau = 10
deta = 2
fai = 1
eta = 100
o = 1
q = 100
e = 1
w = 0.5
b = 5


MEC = [i for i in range(4, 13, 2)]
SE = [1, 3]

class Vehicle(object):
    def __init__(self):

        self.count_step = 0
        self.n_actions = 5

    def reset(self):
        self.count_step = 1
        self.state = np.zeros(shape=(5,2))
        for i in range(5):
            self.state[i][0] = random.choice(MEC)
            self.state[i][1] = random.choice(SE)
        self.BS_list = [BaseStation(self.state[0][0], self.state[0][1]),
                        BaseStation(self.state[1][0], self.state[1][1]),
                        BaseStation(self.state[2][0], self.state[2][1]),
                        BaseStation(self.state[3][0], self.state[3][1]),
                        BaseStation(self.state[4][0], self.state[4][1])]

        return self.state.reshape(-1)

    @property
    def observation_space(self):
        state = np.zeros(shape=(5*2))
        return state

    @property
    def action_space(self):
        # action = np.zeros(5)
        return Box(low = np.float32(np.array([-1,-1,-1,-1,-1])),high = np.float32(np.array([1.0,1.0,1.0,1.0,1.0])),dtype = np.float32)
        # return action

    def step(self, action):
        index = self.decode_vector(action)
        bs = self.BS_list[index]

        # 用户和BS相互映射，这里是每走一步就要进行一下任务卸载，这里的time要根据泊松分布来设置，
        # oUser = OtherUser(bs, time = int(np.random.poisson(1,1)))
        oUser = OtherUser(bs, time = random.randint(10,30)) # 在10~30(都包含)里随机生成

        bs.add_user(oUser)

        # 计算reward
        reward = self.caculate_Reward(bs)

        # all BS user rest_time update。所有bs中的所有用户都-1，因为是虚拟并行操作。可以想成在一秒钟内，所有基站对用户请求都处理了一次，并行执行
        for _bs in self.BS_list:
            for user in list(_bs.users.values()):
                user.time_step()

        # state transition
        for bs in self.BS_list:
            bs.tran_state1()
            bs.tran_state2()

        # next_state
        for i in range(5):
            self.state[i][0] = self.BS_list[i].mec
            self.state[i][1] = self.BS_list[i].se

        done = False

        return self.state.reshape(-1), reward, done, {}


    def sample_action(self):
        indx = random.randint(0, self.n_actions - 1)
        return self.encode_vector(indx, self.n_actions)

    def close(self):
        pass
    def render(self):
        pass
    def caculate_Reward(self, bs):
        reward =  (tau * bs.se * b*(1- w) - deta * b)+ ((fai* bs.mec * o)/(q * eta * q * e))
        return reward

    def encode_vector(self, index, dim):
        vector_encoded = np.random.rand(dim)
        vector_encoded[index] = 1
        return vector_encoded
    def decode_vector(self,vector_decoded):
        vector_decoded_softmax = F.softmax(torch.tensor(vector_decoded), dim=0)
        index = torch.argmax(vector_decoded_softmax, dim=0).item()
        return index


class BaseStation(object):
    def __init__(self, mec, se):
        self.users = {}
        self.user_id = 0
        self.se = int(se)       # the spectrum efficiency
        self.mec = int(mec)    #基站mec的值
        self.ve = 0

    def add_user(self, user):
        self.users[self.user_id] = user
        user.id = self.user_id
        self.user_id += 1
        self.ve = self.ve +1

    def remove_user(self, user):
        self.ve = self.ve - 1
        del self.users[user.id]

    def tran_state1(self):
        if self.mec == 4:
            self.mec = random_pick(MEC, [0.5, 0.25, 0.125, 0.0625, 0.0625])
            return
        elif self.mec == 6:
            self.mec = random_pick(MEC, [0.0625, 0.5, 0.25, 0.125, 0.0625])
            return
        elif self.mec == 8:
            self.mec = random_pick(MEC, [0.0625, 0.0625, 0.5, 0.25, 0.125])
            return
        elif self.mec == 10:
            self.mec = random_pick(MEC, [0.125, 0.0625, 0.0625, 0.5, 0.25])
            return
        elif self.mec == 12:
            self.mec = random_pick(MEC, [0.25, 0.125, 0.0625, 0.0625, 0.5])
            return
        else:
            raise ValueError('Input MEC error!')

    def tran_state2(self):
        if self.se == 1:
            self.se = random_pick(SE, [0.7, 0.3])
            return
        if self.se == 3:
            self.se = random_pick(SE, [0.3, 0.7])
            return
        else:
            raise ValueError('Input v error')

def random_pick(some_list, probability):
    x = random.uniform(0, 1)
    cprobability = 0.0
    global item
    for item, item_probability in zip(some_list, probability):
        cprobability += item_probability
        if x < cprobability:
            break
    return item


class OtherUser(object):
    def __init__(self, bs, time):
        self.bs = bs   # bs和 user相对应
        self.id = 0    #这个是需要的，他会在BS类里进行初始化
        self.rest_time = time   #剩余需要处理的时间

    #  返回的ti 是什么，这个有什么影响吗
    def time_step(self):
        if self.rest_time < 0:
            self.bs.remove_user(self)
        else:
            self.rest_time  = self.rest_time-1


if __name__ == "__main__":
    env = Vehicle()
    state = env.reset()
    for i in range(49):
        action = env.sample_action()
        state, reward, done, info = env.step(action)
        # print(state,reward, action)
