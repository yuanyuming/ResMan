import ray
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


def get_env(average_per_slot=50, machine_num=12):
    env = Environment.VehicleJobSchedulingEnvACE()
    # env = BaseWrapper(env)
    env = PettingZooEnv(env)
    # env = idToAgent(env)
    # env = MultiAgentEnvCompatibility(env)
    return env


def get_env_continuous(
    average_per_slot=50, machine_num=12, allocation_mechanism="FirstPrice"
):
    para = Environment.VehicleJobSchedulingParameters(
        average_per_slot=average_per_slot,
        machine_numbers=machine_num,
        allocation_mechanism=allocation_mechanism,
    )
    para.action_space_continuous = True
    env = Environment.VehicleJobSchedulingEnvACE(parameter=para)
    # env = BaseWrapper(env)
    env = PettingZooEnv(env)
    # env = MultiAgentEnvCompatibility(env)
    return env


if __name__ == "__main__":
    env = get_env_continuous()
    ray.rllib.utils.check_env(env)
