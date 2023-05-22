from gym import Env
from malib.agent import IndependentAgent
from malib.rl.dqn import DEFAULT_CONFIG
from malib.rl.dqn import DQNPolicy, DQNTrainer

algorithms = {
    "default": (
        DQNPolicy,
        DQNTrainer,
        # model configuration, None for default
        {},
        {},
    )
}

trainer_config = DEFAULT_CONFIG["training_config"].copy()
trainer_config["total_timesteps"] = int(1e6)

training_config = {
    "type": IndependentAgent,
    "trainer_config": trainer_config,
    "custom_config": {},
}
from malib.rollout.envs.pettingzoo import PettingZooEnv

import Environment

env = Environment.VehicleJobSchedulingEnv()

