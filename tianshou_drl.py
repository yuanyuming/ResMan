from typing import Any, Dict, Optional, Union

import numpy as np
import supersuit
from pettingzoo.utils.wrappers import BaseWrapper
from tianshou.data import Batch, Collector
from tianshou.env import (
    ContinuousToDiscrete,
    DummyVectorEnv,
    PettingZooEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.utils.net.common import Net

import Environment


def get_env():
    env = Environment.VehicleJobSchedulingEnvACE()
    env = BaseWrapper(env)
    env = PettingZooEnv(env)

    return env


def get_agents(agent_learn, optim):
    env = get_env()
    observation_space = env.observation_space
    net = Net(
        state_shape=observation_space.shape,
        action_shape=env.action_space.shape,
    )


if __name__ == "__main__":
    # 1. Load environment
    env = Environment.VehicleJobSchedulingEnvACE()
    # 2. Wrap the environment with vectorized wrapper
    env = BaseWrapper(env)
    agents = env.num_agents
    env = PettingZooEnv(env)
    policies = MultiAgentPolicyManager([RandomPolicy() for _ in range(10)], env)
    env = SubprocVectorEnv([lambda: env for _ in range(1000)])
    # env = DummyVectorEnv([lambda: env for _ in range(10)])
    # action_space = env.action_spaces[env.agents[0]]
    # 3. Create policy

    # Runtime Environment

    # 4. Create collector
    collector = Collector(policies, env)
    # 5. Execute one episode
    result = collector.collect(n_episode=1000, random=True)
    print(f"Collector return: {result}")
