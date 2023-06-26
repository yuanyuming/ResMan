import torch
from pettingzoo.utils.wrappers import BaseWrapper
from tianshou.env import (
    ContinuousToDiscrete,
    DummyVectorEnv,
    PettingZooEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

from environment import Environment


def get_env():
    env = Environment.VehicleJobSchedulingEnvACE()
    env = BaseWrapper(env)
    env = PettingZooEnv(env)
    return env
