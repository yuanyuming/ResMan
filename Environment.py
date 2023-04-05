import gymnasium as gym
import numpy as np
import Job
import Machine
from gymnasium import spaces


class VehicleJobScheduling(gym.Env):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(self, render_mode=None) -> None:
        self.machines = []
        self.job_generator = Job.JobCollection()
        self.machine_restriction = Machine.MachineRestrict()

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

    def step(self, action):
        observation = 1
        reward = 1
        terminated = False
        info = {}
        return observation, reward, terminated, False, info

    def render(self):
        pass
