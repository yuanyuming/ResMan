from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import DQNPolicy,MultiAgentPolicyManager,RandomPolicy

import Environment
if __name__ == "__main__":
    # 1. Load environment
    env = Environment.VehicleJobSchedulingEnv()
    # 2. Wrap the environment with vectorized wrapper
    env = DummyVectorEnv([lambda: env])
    # 3. Create policy
    policies = MultiAgentPolicyManager([RandomPolicy() for _ in range(env.num_agents)],env)
    # 4. Create collector
    collector = Collector(policies, env)
    # 5. Execute one episode
    result = collector.collect(n_episode=1)
    print(f"Collector return: {result}")

