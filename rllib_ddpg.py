import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

import rllib_setup

ray.init()
env_name = "VJS"
register_env(
    env_name,
    lambda config: rllib_setup.get_env_continuous(),
)
test_env = rllib_setup.get_env_continuous()


def policies(agent_ids):
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    return {
        str(i): (
            None,
            obs_space,
            act_space,
            {}
            # config=config.overrides(agent_id=int(i[8:])),
        )
        for i in agent_ids
    }


config = (
    DDPGConfig()
    .training(lr=0.01)
    .resources(num_gpus=1)
    .multi_agent(
        policies=policies(test_env._agent_ids),
        policy_mapping_fn=lambda agent_id, episode, **kwargs: str(agent_id),
    )
)
config.batch_mode = "complete_episodes"
print(config.to_dict())
# Build a Algorithm object from the config and run one training iteration.
algo = config.build(env=env_name)
algo.train()
