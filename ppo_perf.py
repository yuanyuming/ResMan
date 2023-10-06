import argparse

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

import rllib_setup


def train(jobs, machine, rollout_workers=10):
    env_name = "VJS"
    alg_name = "PPO"
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
        PPOConfig()
        .rollouts(num_rollout_workers=rollout_workers)
        .training(vf_clip_param=1080)
        .resources(num_gpus=2)
        .multi_agent(
            policies=policies(test_env._agent_ids),
            policy_mapping_fn=lambda agent_id, episode, **kwargs: str(agent_id),
        )
        .environment(env=env_name, disable_env_checking=True)
        .evaluation(evaluation_interval=10)
    )
    # print(config.to_dict())
    # Build a Algorithm object from the config and run one training iteration.
    # algo = config.build(env=env_name)

    tune.run(
        alg_name,
        name="PPO" + str(machine) + "_" + str(jobs),
        stop={"episodes_total": 10000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--machine", type=int, default=6)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    train(args.jobs, args.machine, args.workers)
