import functools
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space
from numpy import ndarray
import pettingzoo
from pettingzoo.utils.env import AgentID
import Job
import Machine
import AllocationMechanism
import Auction
from pettingzoo.utils import agent_selector


class VehicleJobSchedulingParameters:
    def __init__(self):
        # Job Config
        self.max_job_vec = [10, 20]
        self.max_job_len = 10

        # Job Distribution Config
        self.job_small_chance = 0.8
        self.job_priority_range = [1, 5]
        self.job_distribution = Job.JobDistribution(
            self.max_job_vec, self.max_job_len, self.job_small_chance
        )
        self.job_dist = self.job_distribution.bi_model_dist
        # Job Collection Config
        self.average_per_slot = 100
        self.duration = 10
        self.max_job_len = 10
        self.job_small_chance = 0.8
        self.job_average_cost_vec = [18, 22]
        self.job_collection = Job.JobCollection(
            self.average_per_slot,
            0,
            0,
            self.duration,
            self.job_dist,
            self.job_distribution.priority_dist,
            self.job_average_cost_vec,
        )

        # Machine Config
        self.machine_numbers = 10
        self.job_backlog_size = 10
        self.job_slot_size = 10
        self.num_res = len(self.max_job_vec)
        self.time_horizon = 10
        self.current_time = 0
        self.machine_average_res_vec = [20, 40]
        self.machine_max_res_vec = [100, 100]
        self.machine_average_cost_vec = [2, 4]

        # Cluster Generate
        self.cluster = Machine.Cluster(
            self.machine_numbers,
            self.job_backlog_size,
            self.job_slot_size,
            self.num_res,
            self.time_horizon,
            self.current_time,
            self.machine_average_res_vec,
            self.machine_average_cost_vec,
        )

        # Machine Restrict Config
        self.max_machines = 5
        self.min_machines = 3

        # Machine Restrict
        self.machine_restrictions = Machine.MachineRestrict(
            self.cluster, self.job_collection, self.max_machines, self.min_machines
        )

        # Job Iterator
        self.job_iterator = Machine.ListIterator(
            iter(self.machine_restrictions))
        # Auction
        self.allocation_mechanism = AllocationMechanism.FirstPrice()
        self.auction_type = Auction.ReverseAuction(
            self.cluster, self.allocation_mechanism
        )

    def reset(self):
        self.machine_restrictions.reset()
        self.job_iterator = Machine.ListIterator(
            iter(self.machine_restrictions))
        self.cluster.reset()


class VehicleJobSchedulingEnv(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        super().__init__()
        self.parameters = parameter
        self.render_mode = render_mode
        self.agents = [
            machine.id for machine in self.parameters.cluster.machines]
        self.possible_agents = self.agents
        self.get_job = self.get_job_next_step()
        self.request_job = None
        self.render_mode = render_mode
        self.total_job = 0
        self.finished_job = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        return spaces.Dict(
            {
                "avail_slot": spaces.MultiDiscrete(self.parameters.machine_max_res_vec),
                "request_job": spaces.Dict(
                    {
                        "len": spaces.Discrete(self.parameters.max_job_len),
                        "res_vec": spaces.MultiDiscrete(self.parameters.max_job_vec),
                        "priority": spaces.Discrete(self.parameters.job_priority_range[1]),
                    }
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        return spaces.Box(low=1 / 3, high=3, shape=(1, 1), dtype=np.float32)

    def reset(self, seed=None, return_info=False, options=None):
        np.random.seed(seed)
        self.parameters.reset()
        self.get_job = self.get_job_next_step()
        self.request_job = None
        self.total_job = 0
        self.finished_job = 0
        observation = {agent: self.parameters.cluster.machines[int(
            agent)].observe() for agent in self.agents}
        return observation

    def render(self):
        print("Current time:", self.parameters.cluster.current_time)
        self.parameters.cluster.show()
        pass

    def get_job_next_step(self):
        for jobs in self.parameters.job_iterator:
            for job in jobs:
                yield job, False
            yield None, True

    def step(self, actions):

        if not actions:
            return {}, {}, {}, {}, {}
        for machine_id, action in actions.items():
            self.parameters.cluster.machines[int(
                machine_id)].action = int(action)
            pass
        if self.request_job is not None:
            self.finished_job += self.parameters.auction_type.auction(
                self.request_job)

        job, done = next(self.get_job)
        self.request_job = job
        if done:
            rw = self.parameters.cluster.step()
            rewards = {agent: rw[int(agent)] for agent in self.agents}

            return {}, rewards, {}, {}, {}
        self.parameters.cluster.clear_job()
        if job is not None:
            self.total_job += 1
            for machine_id in job.restrict_machines:
                self.parameters.cluster.machines[int(
                    machine_id)].request_job = job

        observation = {agent: self.parameters.cluster.machines[int(
            agent)].observe() for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        if self.render_mode == "human":
            self.render()
        return observation, rewards, {}, {}, {}

    def seed(self, seed=None):
        return super().seed(seed)

    def close(self):
        return super().close()

    def state(self) -> ndarray:
        return super().state()


class VehicleJobSchedulingEnvACE(pettingzoo.AECEnv):
    metadata = {"name": "VehicleJobSchedulingEnvACE",
                "render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        super().__init__()
        self.parameters = parameter
        self.render_mode = render_mode
        self.agents = [
            machine.id for machine in self.parameters.cluster.machines]
        self.possible_agents = self.agents
        self.get_job = self.get_job_next_step()
        self.request_job = None
        self.render_mode = render_mode
        self.total_job = 0
        self.finished_job = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        return spaces.Dict(
            {
                "avail_slot": spaces.MultiDiscrete(self.parameters.machine_max_res_vec),
                "request_job": spaces.Dict(
                    {
                        "len": spaces.Discrete(self.parameters.max_job_len),
                        "res_vec": spaces.MultiDiscrete(self.parameters.max_job_vec),
                        "priority": spaces.Discrete(self.parameters.job_priority_range[1]),
                    }
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        return spaces.Box(low=1 / 3, high=3, shape=(1, 1), dtype=np.float32)

    def last(self, observe: bool = True):
        return

    def reset(self, seed=None, return_info=False, options=None):

        np.random.seed(seed)
        self.parameters.reset()
        self.agents = [
            machine.id for machine in self.parameters.cluster.machines]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observation = {agent: self.parameters.cluster.machines[int(
            agent)].observe() for agent in self.agents}
        self.get_job = self.get_job_next_step()
        self.request_job, self.done = next(self.get_job)
        self.total_job = 0
        self.finished_job = 0
        self.__agent_selector = self._agent_selector()
        self.agent_selection = self.__agent_selector.next()

    def render(self):
        print("Current time:", self.parameters.cluster.current_time)
        self.parameters.cluster.show()
        pass

    def get_job_next_step(self):
        for jobs in self.parameters.job_iterator:
            for job in jobs:
                yield job, False
            yield None, True

    def _agent_selector(self):
        if self.request_job is not None:
            return agent_selector(self.request_job.restrict_machines)
        return agent_selector(None)

    def step(self, action):

        if not action:
            return

        agent = self.agent_selection
        self.parameters.cluster.machines[int(agent)].action = int(action)

        if self.__agent_selector.is_last():
            if self.request_job is not None:
                self.finished_job += self.parameters.auction_type.auction(
                    self.request_job)
            self.request_job, self.done = next(self.get_job)

            while not self.done:
                rw = self.parameters.cluster.step()
                self.rewards = {agent: rw[int(agent)] for agent in self.agents}
                self.parameters.cluster.clear_job()
                self.request_job, self.done = next(self.get_job)
                self.observation = {agent: self.parameters.cluster.machines[int(
                    agent)].observe() for agent in self.agents}
                self.request_job, self.done = next(self.get_job)

            if self.request_job is not None:
                self.total_job += 1
                for machine_id in self.request_job.restrict_machines:
                    self.parameters.cluster.machines[int(
                        machine_id)].request_job = self.request_job
                    self.observation[machine_id] = self.parameters.cluster.machines[int(
                        machine_id)].observe()
                self.__agent_selector = self._agent_selector()

        # !TODO update observation,reward
        self.agent_selection = self.__agent_selector.next()
        observation = {agent: self.parameters.cluster.machines[int(
            agent)].observe() for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        if self.render_mode == "human":
            self.render()

    def seed(self, seed=None):
        return np.random.seed(seed)

    def close(self):
        return super().close()

    def state(self) -> ndarray:
        return super().state()
