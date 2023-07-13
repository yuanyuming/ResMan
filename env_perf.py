import time

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import MultiAgentPolicyManager

import tianshou_setup
from static_policy.TruthfulPolicy import TruthfulPolicy

env = tianshou_setup.get_env_continous()


def get_DummyVectorEnv_n(n):
    return DummyVectorEnv([lambda: env for _ in range(n)])


def get_SubprocVectorEnv_n(n):
    return SubprocVectorEnv([lambda: env for _ in range(n)])


def get_policy_manager():
    env = tianshou_setup.get_env_continous()
    policies = MultiAgentPolicyManager([TruthfulPolicy() for _ in range(10)], env)
    return policies


def test_perf():
    for n in range(10, 100, 10):
        print(f"Dummy n={n}")
        env = get_DummyVectorEnv_n(n)
        policies = get_policy_manager()
        collector = Collector(policies, env)
        # Execute one episode
        start = time.time()
        result = collector.collect(n_episode=100)
        end = time.time()
        print(f"Time elapsed: {end-start}")
    for n in range(10, 100, 10):
        print(f"Subproc n={n}")
        env = get_SubprocVectorEnv_n(n)
        policies = get_policy_manager()
        collector = Collector(policies, env)
        # Execute one episode
        start = time.time()
        result = collector.collect(n_episode=100)
        end = time.time()
        print(f"Time elapsed: {end-start}")


if __name__ == "__main__":
    test_perf()
