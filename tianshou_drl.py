from pettingzoo.utils.wrappers import BaseWrapper
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager, RandomPolicy

import Environment

if __name__ == "__main__":
    # 1. Load environment
    env = Environment.VehicleJobSchedulingEnvACE()
    # 2. Wrap the environment with vectorized wrapper
    env = BaseWrapper(env)
    agents = env.num_agents
    env = PettingZooEnv(env)
    # action_space = env.action_spaces[env.agents[0]]
    # 3. Create policy
    policies = MultiAgentPolicyManager([RandomPolicy() for _ in range(10)], env)
    # Runtime Environment
    env = DummyVectorEnv([lambda: env])

    # 4. Create collector
    collector = Collector(policies, env)
    # 5. Execute one episode
    result = collector.collect(n_episode=1)
    print(f"Collector return: {result}")
