"""The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf

Configurations you can try:
    - normal policy gradients (PG)
    - MADDPG
    - QMIX

See also: centralized_critic.py for centralized critic PPO on this game.
"""

import argparse
import logging
import os

import ray
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple
from ray import air, tune
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import register_env
from ray.tune.registry import get_trainable_cls

import rllib_setup

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="MADDPG", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=2000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=700000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=10000, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    env_name = "VJS"
    register_env(
        env_name,
        lambda config: rllib_setup.get_env(),
    )
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=env_name)
        .framework(args.framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    test_env = rllib_setup.get_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policies(agent_ids):
        return {
            str(i): PolicySpec(
                observation_space=obs_space,
                action_space=act_space,
                config=config.overrides(agent_id=int(i[8:])),
            )
            for i in agent_ids
        }

    grouping = {"group": [agent for agent in test_env._agent_ids]}

    if args.run == "MADDPG":
        (
            config.framework("tf")
            .environment(env_config={"actions_are_logits": True})
            .training(num_steps_sampled_before_learning_starts=100)
            .multi_agent(
                policies=policies(test_env._agent_ids),
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: str(
                    agent_id
                ),
            )
        )
    elif args.run == "QMIX":
        (
            config.framework("torch")
            .training(mixer=args.mixer, train_batch_size=32)
            .rollouts(num_rollout_workers=0, rollout_fragment_length=4)
            .exploration(
                exploration_config={
                    "final_epsilon": 0.0,
                }
            )
            .environment(
                env=env_name,
                env_config={
                    "separate_state_space": True,
                    "one_hot_state_encoding": True,
                },
            )
        )

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    results = tune.Tuner(
        args.run,
        run_config=air.RunConfig(stop=stop, verbose=2),
        param_space=config,
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
