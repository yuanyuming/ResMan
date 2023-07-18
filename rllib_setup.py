from pettingzoo.utils.wrappers import BaseWrapper
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from rich import print

from environment import Environment


class idToAgent(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        # self._agent_ids = [i for i in range(len(self._agent_ids))]

    def step(self, action):
        print("The action is", action)
        print("The agent_id is", self.env.agent_selection)
        # print(action[self.env.agent_selection])

        return super().step(action)


def get_env():
    env = Environment.VehicleJobSchedulingEnvACE()
    # env = BaseWrapper(env)
    env = PettingZooEnv(env)
    # env = idToAgent(env)
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
