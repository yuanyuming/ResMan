import dis
import functools
import re
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pettingzoo
import prettytable
from gymnasium import spaces
from gymnasium.spaces import Box, Space
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import ndarray
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID
from rich import print

from . import AllocationMechanism, Auction, Job, Machine


class VehicleJobSchedulingParameters:
    def __init__(
        self,
        distribution="bi_model_dist",
        average_per_slot=10,
        duration=30,
        machine_numbers=12,
        random_cluster=False,
        allocation_mechanism="FirstPrice",
    ):
        # Job Config
        self.max_job_vec = [24, 100]
        self.max_job_len = 10
        self.average_cost_vec = [40.5, 4.5]

        # Job Distribution Config
        self.job_small_chance = 0.8
        self.job_priority_range = [0, 10]
        self.job_distribution = Job.JobDistribution(
            self.max_job_vec,
            self.max_job_len,
            self.job_small_chance,
            self.job_priority_range,
        )
        if distribution == "bi_model_dist":
            self.job_dist = self.job_distribution.bi_model_dist
        else:
            self.job_dist = self.job_distribution.uniform_dist

        # Job Collection Config
        self.average_per_slot = average_per_slot
        self.duration = duration
        self.job_small_chance = 0.8

        self.job_collection = Job.JobCollection(
            self.average_per_slot,
            0,
            0,
            self.duration,
            self.job_dist,
            self.job_distribution.priority_dist,
            self.average_cost_vec,
        )

        # Machine Config
        self.machine_numbers = machine_numbers
        self.num_res = len(self.max_job_vec)
        self.time_horizon = self.max_job_len
        self.current_time = 0
        self.job_backlog_size = 10
        self.job_slot_size = 10
        # Cluster Generate
        self.cluster = Machine.Cluster(
            self.job_backlog_size,
            self.job_slot_size,
            self.num_res,
            self.time_horizon,
            self.current_time,
        )

        if random_cluster:
            self.machine_average_res_vec = [20, 40]
            self.machine_max_res_vec = 100

            self.bias_r = 5
            self.bias_c = 2
            # Add Machine
            self.cluster.generate_machines_random(
                self.machine_numbers,
                self.machine_average_res_vec,
                self.average_cost_vec,
                self.bias_r,
            )

        else:
            self.big_machine = Machine.MachineType(192, 2048, 29.05)
            self.middle_machine = Machine.MachineType(112, 768, 17.45)
            self.small_machine = Machine.MachineType(72, 144, 10.25)
            self.machine_max_res_vec = 4096
            self.machine_types = {
                "small": self.small_machine,
                "middle": self.middle_machine,
                "big": self.big_machine,
            }
            self.cluster.generate_machines_fixed(
                groups=int(self.machine_numbers / 3),
                machine_types=self.machine_types,
                machine_average_cost_vec=self.average_cost_vec,
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
        if allocation_mechanism == "FirstPrice":
            self.allocation_mechanism = AllocationMechanism.FirstPrice()
        elif allocation_mechanism == "SecondPrice":
            self.allocation_mechanism = AllocationMechanism.SecondPrice()
        self.auction_type = Auction.ReverseAuction(
            self.cluster, self.allocation_mechanism
        )
        # observation, action space
        self.action_space_continuous = False
        self.action_discrete_space = 20
        self.action_space_low = 1
        self.action_space_high = 2

        # Runtime Configure
        self.max_step = 100000
        self.total_job = 0
        self.total_job_restrict = 1000
        self.finished_job = 0
        self.finished_job_restrict = 10000
        self.time_step = 0
        self.time_step_restrict = 100
        self.stop_condition = self.stop_condition_time_step

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
                    dtype=np.int16,
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
            "render_modes": ["human", "ascii", "rich_layout"],
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
        self.cnt = 0
        self.max_step = self.parameters.max_step
        self.episode_meta_info = {"max_step": self.max_step}
        self.dones = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.done = False
        self.auction_start = False
        self.round_start = True
        self.round_jobs = None
        self.pay = [0 for _ in range(len(self.agents))]
        self.total_pay = 0
        self.load_blance = 0
        self.obs = spaces.Dict(
            OrderedDict(
                [
                    (
                        "avail_slot",
                        spaces.Box(
                            low=0,
                            high=self.parameters.machine_max_res_vec + 1,
                            shape=(
                                self.parameters.time_horizon,
                                self.parameters.num_res,
                            ),
                            dtype=np.int16,
                        ),
                    ),
                    (
                        "request_res_vec",
                        spaces.Box(
                            low=0,
                            high=np.max(self.parameters.max_job_vec),
                            shape=(self.parameters.num_res,),
                        ),
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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        obs = flatten_space(self.obs)
        return obs

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        if self.parameters.action_space_continuous:
            return Box(
                low=self.parameters.action_space_low,
                high=self.parameters.action_space_high,
                shape=(1,),
                dtype=np.float32,
            )
        return spaces.Discrete(self.parameters.action_discrete_space)

    def reset(self, return_info=False, seed=None, options=None):
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
            agent: flatten(
                self.obs, self.parameters.cluster.machines[int(agent[8:])].observe()
            )
            for agent in self.agents
        }
        self.dones = {agent: False for agent in self.agents}
        self.get_job = self.get_job_next_step()

        self.total_job = 0
        self.finished_job = 0
        self.utility = 0
        self.__agent_selector = self._agent_selector()
        self.agent_selection = self.__agent_selector.next()
        self.round_start = True
        self.round_jobs = None
        self.pay = [0 for _ in range(len(self.agents))]
        self.total_pay = 0
        self.load_blance = 0
        infos = {k: {} for k in self.observation.keys()}
        if return_info:
            return {}
        return self.observation[self.agent_selection], infos

    def observe(self, agent: AgentID) -> Any:
        return self.observation[agent]

    def render(self):
        if self.round_start and self.__agent_selector.is_last():
            table = prettytable.PrettyTable([agent for agent in self.agents])
            table.add_row([self.pay[int(agent[8:])] for agent in self.agents])
            print(table)

        if self.auction_start:
            print(
                self.agent_selection
                + " is biding "
                + str(
                    self.parameters.cluster.machines[
                        int(self.agent_selection[8:])
                    ].get_bid()
                )
            )
        if self.__agent_selector.is_last():
            if self.request_job is not None:
                print(self.request_job.id, " is requesting")
        if self.done:
            self.done = False
            print("Current time:", self.parameters.cluster.current_time)
            if self.round_jobs is not None:
                for job in self.round_jobs:
                    print(job)
            else:
                print("No job this round")
            self.parameters.cluster.show()
        pass

    def get_job_next_step(self):
        for jobs in self.parameters.job_iterator:
            self.round_jobs = jobs
            for job in jobs:
                yield job
            yield None

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
            self.utility += self.request_job.utility

    def next_job(self):
        self.request_job = next(self.get_job)
        if self.request_job is not None:
            self.auction_start = True
            self.total_job += 1
            for machine_id in self.request_job.restrict_machines:
                self.parameters.cluster.machines[
                    int(machine_id)
                ].request_job = self.request_job
                self.observation["Machine_" + str(machine_id)] = flatten(
                    self.obs,
                    self.parameters.cluster.machines[int(machine_id)].observe(),
                )
            self.__agent_selector = self._agent_selector()
        if self.request_job is None:
            self.done = True
            self.auction_start = False
            self.round_end()
            self.observation = {
                agent: flatten(
                    self.obs, self.parameters.cluster.machines[int(agent[8:])].observe()
                )
                for agent in self.agents
            }

            self.__agent_selector = self._agent_selector()
            self.round_start = True

    def round_start_get_pay(self):
        pass

    def round_end(self):
        self.pay = self.parameters.cluster.step()
        self.load_blance += self.parameters.cluster.get_load_balance()
        # print("Current time:", self.parameters.cluster.current_time)
        # self.finished_job = self.parameters.cluster.get_finish_job_total()
        self.parameters.finished_job = self.finished_job
        self.parameters.total_job = self.total_job
        self.parameters.time_step = self.parameters.cluster.current_time
        if self.parameters.stop_condition():
            self.truncations = {agent: True for agent in self.agents}
            self.terminations = {agent: True for agent in self.agents}
            self.done = True
            print(
                "|",
                self.total_job,
                "|",
                self.finished_job,
                "|",
                sum([machine.earning for machine in self.parameters.cluster.machines]),
                "|",
                self.utility,
                "|",
                self.load_blance / self.parameters.cluster.current_time,
                "|",
            )
        self.parameters.cluster.clear_job()

    # @functools.lru_cache(maxsize=1000)
    def action(self, action):
        if self.parameters.action_space_continuous:
            return float(action)
        else:
            return (
                self.parameters.action_space_high - self.parameters.action_space_low
            ) / self.parameters.action_discrete_space * float(
                action
            ) + self.parameters.action_space_low

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            self.agent_selection = self.__agent_selector.next()
            return
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
        # print(agent, "action", action)

        self.parameters.cluster.machines[int(agent[8:])].action = self.action(action)
        # print("Agent:", agent, ", Action:", action)
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
