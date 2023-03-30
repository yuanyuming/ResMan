import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from torch import mode


class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=100, shape=(1,), dtype=float)
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        pass

    def step(self, action):
        self.state += action - 1

        self.shower_length -= 1
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

        pass

    def render(self):
        pass

    def reset(self):
        self.state = np.array([38+random.randint(-3, 3)]).astype(float)
        self.shower_length = 60
        pass


env = ShowerEnv()
log_path = os.path.join('train', 'logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000)
