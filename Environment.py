import gymnasium as gym
import numpy as np
import Job
import Machine
from gymnasium import spaces


class VehicleJobSchedulingParameters:
    def __init__(self):
        # Job Config
        self.max_job_vec = [10, 20]
        self.max_job_len = 10

        # Job Distribution Config
        self.job_small_chance = 0.8
        self.job_distribution = Job.JobDistribution(
            self.max_job_vec, self.max_job_len, self.job_small_chance
        )
        self.job_dist = self.job_distribution.bi_model_dist
        # Job Collection Config
        self.average = 5
        self.duration = 10
        self.max_job_vec = [10, 20]
        self.max_job_len = 10
        self.job_small_chance = 0.8

        self.job_collection = Job.JobCollection(
            self.average, 0, 0, self.duration, self.job_dist
        )

        # Machine Config
        self.machine_numbers = 10
        self.job_backlog_size = 10
        self.job_slot_size = 10
        self.num_res = len(self.max_job_vec)
        self.time_horizon = 20
        self.current_time = 0

        # Cluster Generate
        self.cluster = Machine.Cluster(
            self.machine_numbers,
            self.job_backlog_size,
            self.job_slot_size,
            self.num_res,
            self.time_horizon,
            self.current_time,
        )
        self.cluster.generate_machines_random(self.machine_numbers)

        # Machine Restrict Config
        self.max_machines = 10
        self.min_machines = 3

        # Machine Restrict
        self.machine_restrictions = Machine.MachineRestrict(
            self.cluster, self.job_collection, self.max_machines, self.min_machines
        )

        # Job Iterator
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))
        # Network Config

    def reset(self):
        self.machine_restrictions.reset()
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))


class VehicleJobScheduling(gym.Env):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        self.parameters = parameter

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.parameters.reset()

    def step(self, action):
        # 每Step开始时, 从JobIterator中获取一个JobCollection
        job_list = next(self.parameters.job_iterator)
        # 然后,
        observation = 1
        reward = 1
        terminated = False
        info = {}
        return observation, reward, terminated, False, info

    def render(self):
        pass
