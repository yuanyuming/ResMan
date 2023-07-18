from pettingzoo.utils.wrappers import BaseWrapper
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from environment import Environment


def get_env():
    env = Environment.VehicleJobSchedulingEnvACE()
    # env = BaseWrapper(env)
    env = PettingZooEnv(env)
    # env = MultiAgentEnvCompatibility(env)
    return env


def get_env_continuous():
    para = Environment.VehicleJobSchedulingParameters()
    para.action_space_continuous = True
    env = Environment.VehicleJobSchedulingEnvACE(parameter=para)
    # env = BaseWrapper(env)
    env = PettingZooEnv(env)
    # env = MultiAgentEnvCompatibility(env)
    return env
