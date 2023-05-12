
MOUNTAIN_CAR_CONFIG = "SparseMountainCar-v0"
PENDULUM_CONFIG = "Pendulum-v1"
FROZENLAKE_CONFIG = "FrozenLake-v1"
VEHICLE_CONFIG = "Vehicle"
def print_configs():
    print(f"[{MOUNTAIN_CAR_CONFIG}, {PENDULUM_CONFIG} ,{FROZENLAKE_CONFIG}]")

def get_config(args):

    if args.config_name == PENDULUM_CONFIG:
        config = PendulumConfig()
    elif args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == FROZENLAKE_CONFIG:
        config = FrozenLakeConfig()
    elif args.config_name == VEHICLE_CONFIG:
        config = VehicleConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))
    return config


class Config:
    def __init__(self):
        self.hidden_size = 256
        self.max_episode_len = 300

        self.replay_buffer_size = 1000000
        self.max_frames = 40000
        self.train_epochs = 30   #从原本的500改为了30
        self.batch_size = 128
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.num_steps = 128


        self.automatic_entropy_tuning = True
        self.alpha = 0.2
        self.policy_type = "Gaussian"
        self.lr  = 3e-4

# 这个参数不要再动了，上面的参数也不要动
class PendulumConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v1"
        self.max_frames = 100000


# 先运行一下看具体的情况
class VehicleConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Vehicle"
        self.max_episode_len = 100
        self.max_frames = 100000
        self.train_epochs = 1
        self.num_steps = 1 #原本是1
        self.lr = 3e-4  #默认是很好的
        # self.batch_size = 50


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "SparseMountainCar-v0"
        self.max_frames = 100000
        self.train_epochs = 1
        self.num_steps = 1

class FrozenLakeConfig(Config):
    
    def __init__(self):
        super().__init__()
        self.env_name = "FrozenLake-v1"
        self.max_frames = 100000