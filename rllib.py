"""Uses Ray's RLLib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
import supersuit as ss
from pettingzoo.butterfly import pistonball_v6
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from rllib_setup import get_env_continuous

if __name__ == "__main__":
    ray.init()

    env_name = "VJS"

    register_env(env_name, lambda config: get_env_continuous())
    # ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="tf2")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
