import gymnasium as gym
import numpy as np
from gymnasium import spaces
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
        self.job_distribution = Job.JobDistribution(
            self.max_job_vec, self.max_job_len, self.job_small_chance
        )
        self.job_dist = self.job_distribution.bi_model_dist
        # Job Collection Config
        self.average = 5
        self.duration = 10
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
        self.time_horizon = 10
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
        self.auction = Auction.ReverseAuction(cluster=self.cluster, allocation_mechanism=self.allocation_mechanism)
    def reset(self):
        self.machine_restrictions.reset()
        self.job_iterator = Machine.ListIterator(iter(self.machine_restrictions))


class VehicleJobScheduling(gym.Env):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(
        self, render_mode=None, parameter=VehicleJobSchedulingParameters()
    ) -> None:
        # 初始化环境的参数，可以根据需要添加或修改
        self.num_agents = ...  # 环境中的智能体数量
        self.max_cycles = ...  # 环境的最大步数或时间限制
        self.observation_shape = ...  # 智能体的观测空间形状
        self.action_size = ...  # 智能体的动作空间大小
        # 调用父类的构造方法，传入智能体列表和智能体选择器
        # super().__init__(agents=[f"agent_{i}" for i in range(self.num_agents)],     agent_selector=agent_selector(...=))
        # 初始化智能体的观测空间和动作空间，可以根据需要添加或修改
        # self.observation_spaces = {agent: spaces.Box(low=0, high=255, shape=self.   observation_shape, dtype=np.uint8) for agent in self.agents}
        # self.action_spaces = {agent: spaces.Discrete(self.action_size) for agent in     self.agents}
        self.parameters = parameter

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        #     super().reset(seed=seed)
        #     self.parameters.reset()

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
