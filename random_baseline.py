from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy


import tianshou_setup

env = tianshou_setup.get_env()
policies = MultiAgentPolicyManager(
    [
        RandomPolicy(
            observation_space=env.observation_space, action_space=env.action_space
        )
        for _ in range(len(env.agents))
    ],
    env,
)
# env = SubprocVectorEnv([lambda: env for _ in range(5)])
env = DummyVectorEnv([lambda: env for _ in range(10)])
# action_space = env.action_spaces[env.agents[0]]
# 3. Create policy
# Runtime Environment
# 4. Create collector
collector = Collector(policies, env)
# 5. Execute one episode
result = collector.collect(n_episode=1, random=True)
print(f"Collector return: {result}")
