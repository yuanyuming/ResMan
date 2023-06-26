import timeit
from calendar import c
from typing import Any, Dict, Optional, Union

import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from supersuit import clip_actions_v0
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


def test_env_time():
    for i in range(10):
        i += 1
        envt = SubprocVectorEnv([lambda: env for _ in range(i * 10)])
        collector = Collector(policies, envt)
        start = timeit.default_timer()
        collector.collect(n_episode=100, random=True)
        end = timeit.default_timer()
        print(f"Time taken for {i} SubprocVectorEnv envs: {end - start}")


def test_vec_env_time():
    for i in range(3, 10):
        i += 1
        envd = DummyVectorEnv([lambda: env for _ in range(i * 10)])
        collector = Collector(policies, envd)
        start = timeit.default_timer()
        collector.collect(n_episode=100, random=True)
        end = timeit.default_timer()
        print(f"Time taken for {i} DummyVectorEnv envs: {end - start}")


if __name__ == "__main__":
    # 1. Load environment
    env = Environment.VehicleJobSchedulingEnvACE()
    # 2. Wrap the environment with vectorized wrapper
    env = BaseWrapper(env)
    agents = env.num_agents
    env = PettingZooEnv(env)
    policies = MultiAgentPolicyManager([RandomPolicy() for _ in range(10)], env)
    # env = SubprocVectorEnv([lambda: env for _ in range(5)])
    env = DummyVectorEnv([lambda: env for _ in range(1)])
    # action_space = env.action_spaces[env.agents[0]]
    # 3. Create policy

    # Runtime Environment

    # 4. Create collector
    collector = Collector(policies, env)
    # 5. Execute one episode

    result = collector.collect(n_episode=1)
    print(f"Collector return: {result}")
