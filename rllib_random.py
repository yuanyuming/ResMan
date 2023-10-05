import ray
from ray import tune
from ray.rllib.algorithms.random_agent import RandomAgentConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

import rllib_setup


def train(jobs, machine):
    env_name = "VJS_" + str(machine) + "_" + str(jobs)
    alg_name = "Random"
    register_env(
        env_name,
        lambda config: rllib_setup.get_env_continuous(jobs, machine),
    )
    test_env = rllib_setup.get_env_continuous(jobs, machine)

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
        RandomAgentConfig()
        .rollouts(
            num_rollout_workers=10,
            rollout_fragment_length=30,
        )
        .resources(num_gpus=1)
        .multi_agent(
            policies=policies(test_env._agent_ids),
            policy_mapping_fn=lambda agent_id, episode, **kwargs: str(agent_id),
        )
        .environment(env=env_name, disable_env_checking=True)
    )
    config.batch_mode = "complete_episodes"
    print(config.to_dict())
    # Build a Algorithm object from the config and run one training iteration.
    # algo = config.build(env=env_name)

    algo = config.build()
    print(algo.train())


if __name__ == "__main__":
    ray.init()
    train(50, 12)