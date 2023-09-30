import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env

from rllib_setup import get_env


def train(jobs, machine):
    alg_name = "DQN"
    env_name = "VJS" + str(machine) + "_" + str(jobs)
    register_env(env_name, lambda config: get_env(jobs, machine))

    test_env = get_env(jobs, machine)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policies(agent_ids):
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
        DQNConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=10,
            rollout_fragment_length=30,
            create_env_on_local_worker=True,
            num_envs_per_worker=1,
        )
        .multi_agent(
            policies=policies(test_env._agent_ids),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
        .evaluation(evaluation_interval=10)
    )

    tune.run(
        alg_name,
        name="DQN" + str(machine) + "_" + str(jobs),
        stop={"episodes_total": 10000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )
