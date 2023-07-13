from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager

import tianshou_setup
from static_policy.TruthfulPolicy import TruthfulPolicy

env = tianshou_setup.get_env_continous()
policies = MultiAgentPolicyManager([TruthfulPolicy() for _ in range(10)], env)

env = DummyVectorEnv([lambda: env for _ in range(1)])

collector = Collector(policies, env)
# Execute one episode
result = collector.collect(n_episode=10)
print(f"Collector return: {result}")
