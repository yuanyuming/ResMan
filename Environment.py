import functools
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from urllib import request

import gymnasium as gym
import numpy as np
import pettingzoo
import prettytable
from gymnasium import spaces
from gymnasium.spaces import Space
from numpy import ndarray
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID

import AllocationMechanism
import Auction
import Job
import Machine


class VehicleJobSchedulingParameters:
    def __init__(self):
        # Job Config
        self.max_job_vec = [20, 30]
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
        self.duration = 30
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
        self.machine_max_res_vec = 100
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
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))
        # Auction
        self.allocation_mechanism = AllocationMechanism.FirstPrice()
        self.auction_type = Auction.ReverseAuction(
            self.cluster, self.allocation_mechanism
        )

        # Runtime Configure
        self.total_job = 0
        self.total_job_restrict = 10000
        self.finished_job = 0
        self.finished_job_restrict = 10000
        self.time_step = 0
        self.time_step_restrict = 1000
        self.stop_condition = self.stop_condition_total_job

    def reset(self):
        """
        Reset the environment
        """
        self.machine_restrictions.reset()
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))
        self.cluster.reset()

    def stop_condition_total_job(self):
        """
        Check if the total job is enough
        """

        return self.total_job >= self.total_job_restrict

    def stop_condition_finished_job(self):
        """
        Check if the finished job is enough
        """
        self.finished_job = self.cluster.get_finish_job_total()
        return self.finished_job >= self.finished_job_restrict

    def stop_condition_time_step(self):
        """
        Check if the time step is enough
        """
        self.time_step = self.cluster.current_time
        return self.time_step >= self.time_step_restrict


class VehicleJobSchedulingEnv(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        super().__init__()
        self.parameters = parameter
        self.render_mode = render_mode
        self.agents = [
            "Machine_" + str(machine.id) for machine in self.parameters.cluster.machines
        ]
        self.possible_agents = self.agents
        self.get_job = self.get_job_next_step()
        self.request_job = None
        self.render_mode = render_mode
        self.total_job = 0
        self.finished_job = 0
        self.round_start = True

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        return spaces.Dict(
            {
                "avail_slot": spaces.Box(
                    low=0,
                    high=self.parameters.machine_max_res_vec,
                    shape=(self.parameters.num_res, self.parameters.time_horizon),
                    dtype=np.int8,
                ),
                "request_job": spaces.Dict(
                    {
                        "len": spaces.Discrete(self.parameters.max_job_len),
                        "res_vec": spaces.MultiDiscrete(self.parameters.max_job_vec),
                        "priority": spaces.Discrete(
                            self.parameters.job_priority_range[1]
                        ),
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
        observation = {
            agent: self.parameters.cluster.machines[int(agent[8:])].observe()
            for agent in self.agents
        }
        self.round_start = True
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
        for machine, action in actions.items():
            self.parameters.cluster.machines[int(machine[8:])].action = int(action)
            pass
        if self.request_job is not None:
            self.finished_job += self.parameters.auction_type.auction(self.request_job)

        job, done = next(self.get_job)
        self.request_job = job
        if done:
            rw = self.parameters.cluster.step()
            rewards = {agent: rw[int(agent[8:])] for agent in self.agents}

            return {}, rewards, {}, {}, {}
        self.parameters.cluster.clear_job()
        if job is not None:
            self.total_job += 1
            for machine_id in job.restrict_machines:
                self.parameters.cluster.machines[int(machine_id)].request_job = job

        observation = {
            agent: self.parameters.cluster.machines[int(agent[8:])].observe()
            for agent in self.agents
        }
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
    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        super().__init__()
        self.metadata = {
            "name": "VehicleJobSchedulingEnvACE",
            "render_modes": ["human", "ascii"],
        }
        self.parameters = parameter
        self.render_mode = render_mode
        self.agents = [
            "Machine_" + str(machine.id) for machine in self.parameters.cluster.machines
        ]
        self.possible_agents = self.agents
        self.get_job = self.get_job_next_step()
        self.request_job = None
        self.render_mode = render_mode
        self.total_job = 0
        self.finished_job = 0
        self.observation = {
            agent: self.parameters.cluster.machines[int(agent[8:])].observe()
            for agent in self.agents
        }
        self.dones = {agent: False for agent in self.agents}
        self.done = False
        self.auction_start = False
        self.round_start = True
        self.round_jobs = None
        self.pay = [0 for _ in range(len(self.agents))]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        obs = spaces.Dict(
            OrderedDict(
                [
                    (
                        "avail_slot",
                        spaces.Box(
                            low=0,
                            high=self.parameters.machine_max_res_vec,
                            shape=(
                                self.parameters.time_horizon,
                                self.parameters.num_res,
                            ),
                            dtype=np.int8,
                        ),
                    ),
                    (
                        "request_res_vec",
                        spaces.MultiDiscrete(self.parameters.max_job_vec),
                    ),
                    (
                        "request_len",
                        spaces.Discrete(self.parameters.max_job_len),
                    ),
                    (
                        "request_priority",
                        spaces.Discrete(self.parameters.job_priority_range[1] + 1),
                    ),
                ]
            )
        )
        return spaces.Dict({"observation": obs})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        return spaces.Box(low=1 / 3, high=3, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, return_info=False, options=None):
        np.random.seed(seed)
        self.parameters.reset()
        self.agents = [
            "Machine_" + str(machine.id) for machine in self.parameters.cluster.machines
        ]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observation = {
            agent: self.parameters.cluster.machines[int(agent[8:])].observe()
            for agent in self.agents
        }
        self.dones = {agent: False for agent in self.agents}
        self.get_job = self.get_job_next_step()

        self.total_job = 0
        self.finished_job = 0
        self.__agent_selector = self._agent_selector()
        self.agent_selection = self.__agent_selector.next()
        self.round_start = True
        self.round_jobs = None
        self.pay = [0 for _ in range(len(self.agents))]
        return self.observation[self.agent_selection], {}

    def observe(self, agent: AgentID) -> Any:
        return {"observation": self.observation[agent]}

    def render(self):
        if self.round_start and self.__agent_selector.is_last():
            table = prettytable.PrettyTable([agent for agent in self.agents])
            table.add_row([self.pay[int(agent[8:])] for agent in self.agents])
            print(table)

        if self.auction_start:
            print(self.agent_selection + " is biding")
        if self.__agent_selector.is_last():
            print(self.request_job)
        if self.done:
            print("Current time:", self.parameters.cluster.current_time)
            if self.round_jobs is not None:
                for job in self.round_jobs:
                    print(job)
            else:
                print("No job this round")
            self.parameters.cluster.show()
            print(self._cumulative_rewards)
        pass

    def get_job_next_step(self):
        for jobs in self.parameters.job_iterator:
            self.round_jobs = jobs
            for job in jobs:
                yield job, False
            yield None, True

    def _agent_selector(self):
        if self.request_job is not None:
            return agent_selector(
                [
                    "Machine_" + str(machine_id)
                    for machine_id in self.request_job.restrict_machines
                ]
            )
        return agent_selector(self.possible_agents)

    def auction(self):
        if self.request_job is not None:
            self.finished_job += self.parameters.auction_type.auction(self.request_job)

    def next_job(self):
        self.request_job, self.done = next(self.get_job)
        if self.done:
            self.auction_start = False
            self.round_end()
            self.observation = {
                agent: self.parameters.cluster.machines[int(agent[8:])].observe()
                for agent in self.agents
            }

            self.__agent_selector = self._agent_selector()
            self.round_start = True

        if self.request_job is not None:
            self.auction_start = True
            self.total_job += 1
            for machine_id in self.request_job.restrict_machines:
                self.parameters.cluster.machines[
                    int(machine_id)
                ].request_job = self.request_job
                self.observation[machine_id] = self.parameters.cluster.machines[
                    int(machine_id)
                ].observe()
            self.__agent_selector = self._agent_selector()

    def round_start_get_pay(self):
        pass

    def round_end(self):
        self.pay = self.parameters.cluster.step()
        # print("Current time:", self.parameters.cluster.current_time)
        self.finished_job = self.parameters.cluster.get_finish_job_total()
        self.parameters.finished_job = self.finished_job
        self.parameters.total_job = self.total_job
        self.parameters.time_step = self.parameters.cluster.current_time
        if self.parameters.stop_condition():
            self.terminations = {agent: True for agent in self.agents}
            # print("Finished!!!")
        self.parameters.cluster.clear_job()

    def step(self, action):
        agent = self.agent_selection
        if not self.round_start:
            self._clear_rewards()
            self._cumulative_rewards[agent] = 0
        if self.round_start:
            # print(agent, "get pay", self.pay[int(agent[8:])])
            self._clear_rewards()
            self.rewards[agent] = self.pay[int(agent[8:])]
            self._cumulative_rewards[agent] = 0
            self._accumulate_rewards()
        if self.round_start and self.__agent_selector.is_last():
            self.round_start = False

        self.parameters.cluster.machines[int(agent[8:])].action = float(action)

        if self.__agent_selector.is_last():
            # self._clear_rewards()
            # print(self.request_job)
            self.auction()

            self.next_job()

        # !TODO update observation,reward
        self.agent_selection = self.__agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def seed(self, seed=None):
        return np.random.seed(seed)

    def close(self):
        return

    def state(self) -> ndarray:
        return super().state()
