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
        self.job_average_cost_vec = [8, 12]
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

    def reset(self):
        self.machine_restrictions.reset()
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))


class VehicleJobSchedulingEnv(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        super().__init__()
        self.parameters = parameter
        self.render_mode = render_mode
        self.agents = [machine.id for machine in self.parameters.cluster.machines]
        self.possible_agents = self.agents

    def observation_space(self, agent: AgentID) -> Space:
        return spaces.Dict(
            {
                "avail_slot": spaces.MultiDiscrete(self.parameters.cluster.machines[int(agent)].res_slot_time),
                "request_job": spaces.Dict(
                    {
                        "len": spaces.Discrete(self.parameters.max_job_len),
                        "res_vec": spaces.MultiDiscrete(self.parameters.max_job_vec),
                        "priority": spaces.Discrete(self.parameters.job_priority_range[1]),
                    }
                ),
            }
        )

    def action_space(self, agent: AgentID) -> Space:
        return spaces.Box(low=1 / 3, high=3, shape=(1, 1), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.parameters.reset()

        # def step(self, action):
        #     # 检查当前智能体是否有效
        #     #if self.dones[self.agent_selection]:
        #         return self._was_done_step(action)
        #     # 检查动作是否合法
        #     #if not self.action_spaces[self.agent_selection].contains(action):
        #         raise ValueError(
        #             f"Invalid action {action} for agent {self. agent_selection}"
        #         )
        #     # 更新环境的状态和智能体的观测、奖励等变量
        #     # 根据环境的逻辑和规则，编写相应的代码
        #     # ...
        #     # 设置终止和截断标志
        #     #self.dones[self.agent_selection] = ...  # 根据环境的逻辑和规则，判断当前智能体是否   完成
        #     #self.terminal = ...  # 根据环境的逻辑和规则，判断整个环境是否结束
        #     #self.truncation = ...  # 根据环境的逻辑和规则，判断是否达到最大步数或时间限制
        #     # 如果有必要，调用self._was_dead_step(action)函数来处理智能体的死亡或游戏结束的情况
        #     #if self.dones[self.agent_selection] or self.terminal:
        #         return self._was_dead_step(action)
        #     # 切换到下一个智能体
        #     #self._cumulative_rewards[self.agent_selection] = 0
        #     #self.agent_selection = self._agent_selector.next()
        #     # 每Step开始时, 从JobIterator中获取一个JobCollection
        #     #job_list = next(self.parameters.job_iterator)
        #     # 然后,
        #     observation = 1
        #     reward = 1
        #     terminated = False
        #     info = {}
        #     return observation, reward, terminated, False, info
        pass

    def render(self):
        pass

    def step(self, actions):
        
        if not actions:
            return {},{},{},{},{}
        for machine_id, action in actions.items():
            self.parameters.cluster.machines[int(machine_id)].action = action
            pass
        
        jobs = next(self.parameters.job_iterator)
        pass

    def seed(self, seed=None):
        return super().seed(seed)

    def close(self):
        return super().close()

    def state(self) -> ndarray:
        return super().state()
