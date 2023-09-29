"""
定义任务Job相关结构
"""
# 定义一个类，表示任务分配的参数
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 导入numpy库
import jax.numpy as np
import pettingzoo
import prettytable
from gymnasium import spaces
from gymnasium.spaces import Box, Space
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import ndarray
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID
from rich import print
from scipy.stats import poisson


class JobDistribution:
    # 初始化方法，接受以下参数：
    # max_job_vec: 一个列表，表示每种资源的最大任务数量
    # max_job_len: 一个整数，表示任务的最大时长
    # job_small_chance: 一个浮点数，表示任务是小任务的概率
    # job_priority_range: 一个列表，表示任务的优先级范围
    def __init__(
        self,
        max_job_vec: List[int] = [10, 20],
        max_job_len: int = 10,
        job_small_chance: float = 0.8,
        job_priority_range: List[int] = [1, 5],
    ):
        self.num_res = len(max_job_vec)  # 资源的种类数
        self.max_nw_size = max_job_vec  # 每种资源的最大任务数量
        self.max_job_len = max_job_len  # 任务的最大时长

        self.job_small_chance = job_small_chance  # 任务是小任务的概率

        self.job_len_big_lower = int(max_job_len * 2 / 3)  # 大任务的最小时长
        self.job_len_big_upper = max_job_len  # 大任务的最大时长

        self.job_len_small_lower = 1  # 小任务的最小时长
        self.job_len_small_upper = int(max_job_len / 5)  # 小任务的最大时长

        self.dominant_res_lower = np.divide(np.array(max_job_vec), 2)  # 占主导地位的资源的最小请求量
        self.dominant_res_upper = max_job_vec  # 占主导地位的资源的最大请求量

        self.other_res_lower = 1  # 其他资源的最小请求量
        self.other_res_upper = np.divide(np.array(max_job_vec), 5)  # 其他资源的最大请求量
        self.priority_range = job_priority_range  # 任务的优先级范围

    # 定义一个方法，返回类的字符串表示
    def __str__(self) -> str:
        return f"JobDistribution({self.max_nw_size}, {self.max_job_len}, {self.job_small_chance})"

    # 定义一个方法，返回一个随机生成的任务优先级
    def priority_dist(self):
        return np.random.randint(self.priority_range[0], self.priority_range[1] + 1)

    # 定义一个方法，返回一个正态分布生成的任务时长和资源请求量
    def uniform_dist(self):
        # NOTE - 新任务时长
        nw_len = np.random.randint(1, self.max_job_len + 1)  # 随机生成一个整数作为任务时长

        nw_size = np.zeros(self.num_res)  # 创建一个零向量作为资源请求量

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size[i] + 1)  # 随机生成每种资源的请求量

        return nw_len, nw_size  # 返回任务时长和资源请求量

    # 定义一个方法，返回一个双峰分布生成的任务时长和资源请求量
    def bi_model_dist(self):
        # NOTE - 新任务时长
        if np.random.rand() < self.job_small_chance:  # 如果随机数小于小任务概率，则生成一个小任务
            nw_len = np.random.randint(
                self.job_len_small_lower, self.job_len_small_upper + 1
            )  # 随机生成一个整数作为小任务时长
        else:  # 否则生成一个大任务
            nw_len = np.random.randint(
                self.job_len_big_lower, self.job_len_big_upper
            )  # 随机生成一个整数作为大任务时长

        # NOTE - 任务资源请求量
        dominant_res = np.random.randint(0, self.num_res)  # 随机选择一种资源作为占主导地位的资源
        nw_size = np.zeros([self.num_res])  # 创建一个零向量作为资源请求量

        for i in range(self.num_res):
            if i == dominant_res:  # 如果是占主导地位的资源，则生成较高的请求量
                nw_size[i] = np.random.randint(
                    self.dominant_res_lower[i], self.dominant_res_upper[i] + 1
                )
            else:  # 如果是其他资源，则生成较低的请求量
                nw_size[i] = np.random.randint(
                    self.other_res_lower, self.other_res_upper[i] + 1
                )

        return nw_len, nw_size  # 返回任务时长和资源请求量


# 定义一个类，表示一个任务


class Job:
    """
    res_vec:资源需求向量
    job_len:任务长度
    job_id:任务id,唯一
    enter_time:进入队列的时间
    # TODO - Job 直接传参Dist
    """

    # 初始化方法，接受以下参数：
    # res_vec: 一个列表，表示任务需要的每种资源的数量
    # job_len: 一个整数，表示任务的时长
    # job_id: 一个整数，表示任务的唯一标识符
    # enter_time: 一个整数，表示任务进入队列的时间
    # priority: 一个整数，表示任务的优先级
    # average_cost_vec: 一个列表，表示每种资源的平均成本
    def __init__(
        self,
        res_vec: List[int] = [1, 1],
        job_len: int = 1,
        job_id: int = 0,
        enter_time: int = 0,
        priority: int = 0,
        average_cost_vec: List[int] = [1, 10],
    ):
        self.id = job_id  # 任务id
        self.res_vec = res_vec  # 资源需求向量
        self.len = job_len  # 任务长度
        self.restrict_machines = [1, 2]  # 限制的机器列表
        self.running_machine = 0  # 运行的机器编号
        self.enter_time = enter_time  # 进入队列的时间
        self.time_restrict = 0  # 时间限制
        self.start_time = -1  # 开始时间
        self.finish_time = -1  # 结束时间
        self.priority = priority  # 优先级
        # self.job_vec = self.generate_job()  # 生成的任务向量
        self.average_cost_vec = average_cost_vec  # 平均成本向量
        self.budget = self.calculate_budget(average_cost_vec)  # 计算的预算
        self.pay = 0  # 支付金额
        self.utility = 0  # 效用值

    # 定义一个方法，根据给定的分布函数随机生成任务时长和资源需求向量，并更新预算和任务向量
    def random_job(self, dist=JobDistribution().bi_model_dist) -> None:
        self.len, self.res_vec = dist()  # 调用分布函数生成时长和需求向量
        self.budget = self.calculate_budget(self.average_cost_vec)  # 更新预算
        # self.job_vec = self.generate_job()  # 更新任务向量

    # 定义一个方法，根据资源需求向量生成任务向量，即每个时刻需要的资源数量
    def generate_job(self):
        return [self.res_vec] * self.len

    # 定义一个方法，根据每种资源的平均成本和方差计算任务的预算，并保证预算不为负数
    def calculate_budget(self, average_cost_vec, var=0.03):
        return max(
            0,
            np.dot(np.array(self.res_vec), np.array(average_cost_vec))
            * self.len
            * ((1 + self.priority / 10) + var * np.random.normal()),
        )

    # 定义一个方法，返回任务的观察信息，即资源需求向量、时长、优先级和限制机器列表
    def observe(self):
        job_obs = OrderedDict(
            {"res_vec": self.res_vec, "len": self.len, "priority": self.priority}
        )
        return job_obs

    # 定义一个方法，返回任务的请求信息，即id、资源需求向量、时长、优先级和限制机器列表
    def request(self):
        return self.id, self.res_vec, self.len, self.priority, self.restrict_machines

    # 定义一个方法，设置任务的开始时间，并根据时长计算结束时间，并更新效用值为预算减去支付金额除以时长乘以优先级（暂定）
    def start(self, start_time):
        self.start_time = start_time
        self.finish(start_time + self.len)  # 设置结束时间
        self.utility = (self.budget - self.pay) / self.len * self.priority  # 更新效用值

    # 定义一个方法，设置任务的结束时间
    def finish(self, finish_time):
        self.finish_time = finish_time

    # 定义一个方法，将任务信息转换为列表形式（用于打印）
    def to_list(self):
        return [
            self.id,
            self.res_vec,
            self.len,
            self.priority,
            self.budget,
            self.restrict_machines,
            self.enter_time,
            self.start_time,
            self.finish_time,
            self.utility,
        ]

    def static_info(self):
        return {
            "id": str(self.id),
            "res_vec": str(self.res_vec),
            "len": str(self.len),
            "priority": str(self.priority),
            "budget": f"{self.budget:.2f}",
            "restrict_machines": str(self.restrict_machines),
        }

    def info(self):
        return {
            "pay": self.pay,
            "utility": self.utility,
        }

    # 定义一个方法，打印出任务信息（用于展示）
    def show(self):
        table = prettytable.PrettyTable(
            [
                "Job Id",
                "Res Vector",
                "Job Len",
                "Priority",
                "Budget",
                "Restrict Machines",
                "Enter Time",
                "Start Time",
                "Finish Time",
                "Utility",
            ]
        )
        table.add_row(self.to_list())
        table.set_style(prettytable.MSWORD_FRIENDLY)
        table.title = "Job Info"
        print(table)
        print("Job Vector")
        print(self.generate_job())

    # 定义一个方法，设置任务的支付金额（暂未实现）
    def get_pay(self, pay=0):
        self.pay = pay

    # 定义一个方法，返回任务的字符串表示（用于调试）
    def __str__(self):
        return "Job id:{},Res Vector:{},Job Len:{},Restrict Machine:{},Budget:{},Pay:{},Running:{}".format(
            self.id,
            self.res_vec,
            self.len,
            self.restrict_machines,
            self.budget,
            self.pay,
            self.running_machine,
        )


class JobCollection:
    def __init__(
        self,
        average=5,
        id_start=0,
        enter_time=0,
        duration=10,
        job_dist=JobDistribution().bi_model_dist,
        job_priority_dist=JobDistribution().priority_dist,
        averge_cost_vec=[4, 6],
    ):
        self.average = average
        self.id_start = id_start
        self.enter_time = enter_time
        self.Dist = job_dist
        self.priority = job_priority_dist
        self.now_id = id_start
        self.duration = duration
        self.average_cost_vec = averge_cost_vec

    def reset(self):
        self.now_id = self.id_start
        self.enter_time = 0

    def get_job_collection(self):
        """
        Purpose:
        """
        poi = poisson.rvs(self.average)
        collection = []
        for id in range(self.now_id, self.now_id + poi):
            job_len, job_res_vec = self.Dist()
            job = Job(
                job_res_vec,
                job_len,
                id,
                self.enter_time,
                self.priority(),
                self.average_cost_vec,
            )
            collection.append(job)
        self.enter_time += 1
        self.now_id += poi
        return collection

    def __str__(self):
        return "id_start:{},enter_time:{},now_id:{},duration:{}".format(
            self.id_start, self.enter_time, self.now_id, self.duration
        )

    def get_job_collections(self):
        """
        Purpose:
        """
        poi = poisson.rvs(self.average, size=self.duration)
        collection = []
        collections = []
        for t in range(self.duration):
            for id in range(self.now_id, self.now_id + int(poi[t])):
                job_len, job_res_vec = self.Dist()
                job = Job(
                    job_res_vec,
                    job_len,
                    id,
                    self.enter_time,
                    self.priority(),
                    self.average_cost_vec,
                )
                collection.append(job)
            self.enter_time += 1
            self.now_id += poi[t]
            collections.append(collection)
            collection = []
        return collections

    def __iter__(self):
        return self

    def __next__(self):
        collections = self.get_job_collections()
        return collections


class JobSlot:
    """
    Define the JobSlot.
    """

    def __init__(self, num_nw):
        self.slot = [None] * num_nw
        self.surplus_slot = num_nw

    def show(self):
        """
        Purpose: show the JobSlot
        """
        table = prettytable.PrettyTable(
            [
                "Job Id",
                "Res Vector",
                "Job Len",
                "Enter Time",
                "Start Time",
                "Finish Time",
            ]
        )
        for job in self.slot:
            if job is not None:
                table.add_row(
                    [
                        job.id,
                        job.generate_job(),
                        job.len,
                        job.enter_time,
                        job.start_time,
                        job.finish_time,
                    ]
                )
            else:
                table.add_row([None] * 6)
        table.title = "JobSlot Info"
        print(table)
        print("Surplus Slot:{}".format(self.surplus_slot))

    def add_new_job(self, job):
        """
        Purpose:
        """
        for i in range(len(self.slot)):
            if self.slot[i] is None:
                self.slot[i] = job
                self.surplus_slot -= 1
                break

    def select_job(self, num=0):
        job = self.slot[num]
        self.slot[num] = None
        self.surplus_slot += 1
        return job

    def __str__(self):
        table = prettytable.PrettyTable(
            [
                "Job Id",
                "Res Vector",
                "Job Len",
                "Enter Time",
                "Start Time",
                "Finish Time",
            ]
        )
        for job in self.slot:
            if job is not None:
                table.add_row(
                    [
                        job.id,
                        job.generate_job(),
                        job.len,
                        job.enter_time,
                        job.start_time,
                        job.finish_time,
                    ]
                )
            else:
                table.add_row([None] * 6)
        table.title = "JobSlot Info"
        return str(table) + "\nSurplus Slot:{}".format(self.surplus_slot)


class JobBacklog:
    """
    Backlog of the jobs
    """

    def __init__(self, backlog_size) -> None:
        self.backlog_size = backlog_size
        self.backlog = [None] * backlog_size
        self.curr_size = 0

    def show(self):
        """
        Purpose: show the JobBacklog
        """
        table = prettytable.PrettyTable(
            ["Job Id", "Res Vector", "Enter Time", "Start Time", "Finish Time"]
        )
        for job in self.backlog:
            if job is not None:
                table.add_row(
                    [
                        job.id,
                        job.generate_job(),
                        job.enter_time,
                        job.start_time,
                        job.finish_time,
                    ]
                )

        table.title = "JobBacklog"
        print(table)

    def add_job(self, job):
        """
        Purpose:
        """
        if self.curr_size < self.backlog_size:
            self.backlog[self.curr_size] = job
            self.curr_size += 1
        else:
            self.backlog[:-1] = self.backlog[1:]
            self.backlog[-1] = job


class JobRecord:
    def __init__(self) -> None:
        self.record = {}

    def new_record(self, job=Job()):
        """
        Purpose: add new record
        """
        self.record[job.id] = job

    def show(self):
        """
        Purpose:
        """
        table = prettytable.PrettyTable(
            ["Job Id", "Res Vector", "Enter Time", "Start Time", "Finish Time"]
        )
        for record in self.record:
            table.add_row(
                [
                    self.record[record].id,
                    self.record[record].res_vec,
                    self.record[record].enter_time,
                    self.record[record].start_time,
                    self.record[record].finish_time,
                ]
            )
        table.title = "JobRecord"
        print(table)







class SlotShow:
    def __init__(
        self,
        res_slot=[10, 15],
        avial_slot=[[0, 1], [2, 1], [7, 10], [0, 1], [2, 1], [7, 10]],
    ):
        self.res_slot = np.asarray(res_slot)  # convert to numpy array for vectorization
        self.avail_slot = np.asarray(avial_slot)
        # use vectorize function to apply the calculation to each element
        # generate bars using list comprehension and store it as a class attribute
        self.bars = [char for char in " ▁▂▃▄▅▆▇█"]
        self.percent_slot = np.vectorize(lambda x, y: int(x / y * 8))(
            self.avail_slot, self.res_slot
        )

    def compute_chart(self, avail_slot):
        self.avail_slot = np.asarray(avail_slot)
        self.percent_slot = np.vectorize(lambda x, y: int(x / y * 8))(
            self.avail_slot, self.res_slot
        )
        for i in range(len(self.res_slot)):
            bar_show = [self.bars[s] for s in self.percent_slot[:, i]]
            # use f-string to format the output
            print(self.percent_slot[:, i])
            print(f"- Resources #{i}:𝄃{''.join(bar_show)}𝄃")

    def compute_dict(self, avail_slot):
        self.avail_slot = np.asarray(avail_slot)
        self.percent_slot = np.vectorize(lambda x, y: int(x / y * 8))(
            self.avail_slot, self.res_slot
        )
        # use a dictionary comprehension to generate a dictionary with the same functionality as compute_chart
        return {
            f"Res {i}": "".join([self.bars[s] for s in self.percent_slot[:, i]])
            for i in range(len(self.res_slot))
        }


class BiderPolicy:
    def __init__(self) -> None:
        pass

    def bid(self, job=Job.Job()):
        pass


@dataclass
class MachineType:
    vCPUs: int
    memory: int
    price: float


class Machine:
    def __init__(
        self,
        id=0,
        num_res=2,
        time_horizon=20,
        job_slot_size=10,
        job_backlog_size=10,
        res_slot=[20, 40],
        cost_vector=[4, 6],
        current_time=0,
        price=0.0,
    ) -> None:
        """
        Initializes the machine
        """
        self.id = id
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time
        self.job_slot_size = job_slot_size
        self.job_backlog_size = job_backlog_size
        self.price = price
        self.job_slot = Job.JobSlot(self.job_slot_size)
        self.job_backlog = Job.JobBacklog(self.job_backlog_size)
        self.res_slot = res_slot
        self.reward = 0
        self.earning = 0
        self.finished_job = []
        self.finished_job_num = 0
        self.cost_vector = cost_vector
        self.avail_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot
        self.res_slot_time = self.avail_slot
        self.running_job = []
        # Bid
        self.request_job = None
        self.policy = self.drl_bid
        self.action = 1
        self.bid = 0
        self.slot_show = SlotShow(self.res_slot, self.avail_slot)

    def add_backlog(self, job=Job.Job()):
        """
        Add the Job to the backlog
        """
        self.job_backlog.add_job(job)

    def reset(self):
        self.job_slot = Job.JobSlot(self.job_slot_size)
        self.job_backlog = Job.JobBacklog(self.job_backlog_size)
        self.earning = 0
        self.finished_job = []
        self.avail_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot
        self.running_job = []
        self.request_job = None

    def get_bid(self):
        if self.request_job is not None:
            self.bid = self.policy(self.request_job)
            return self.bid
        return 0

    def drl_bid(self, job=Job.Job()):
        return self.action * (np.dot(job.res_vec, self.cost_vector)) * job.len

    def set_action(self, action=1):
        self.action = action

    def fixed(self, job=Job.Job()):
        return np.dot(job.res_vec, self.cost_vector)

    def clear_job(self):
        self.request_job = None

    def observe(self):
        if self.request_job is not None:
            machine_obs = OrderedDict(
                [
                    ("avail_slot", np.array(self.avail_slot, dtype=np.int16)),
                    (
                        "request_res_vec",
                        np.array(self.request_job.res_vec),
                    ),
                    (
                        "request_len",
                        self.request_job.len,
                    ),
                    (
                        "request_priority",
                        self.request_job.priority,
                    ),
                ]
            )
            return machine_obs
        machine_obs = OrderedDict(
            [
                ("avail_slot", np.array(self.avail_slot, dtype=np.int16)),
                (
                    "request_res_vec",
                    np.array([0, 0]),
                ),
                (
                    "request_len",
                    0,
                ),
                (
                    "request_priority",
                    0,
                ),
            ]
        )
        return machine_obs

    def fixed_norm(self, job=Job.Job(), var=0.2):
        return (
            (np.dot(job.res_vec, self.cost_vector) + var * np.random.normal())
            * job.len
            * job.priority
        )

    def request_auction(self, job=Job.Job()):
        self.request_job = job

    # async allocate_job, not used

    def can_allocate(self, job=Job.Job()):
        """
        Check if the Job can be allocated to this Machine
        """
        allocated = False
        new_avail_res = self.avail_slot[0 : job.len, :] - job.res_vec
        if np.all(new_avail_res[:] >= 0):
            allocated = True
        return allocated

    # async allocate_job, not used

    def can_allocate_async(self, job=Job.Job()):
        """
        Check if the Job can be allocated to this Machine
        """
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i : i + job.len, :] - job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True
                break
        return allocated

    def allocate_job(self, job=Job.Job()):
        """
        Allocate the Job to this Machine
        """
        allocated = False
        new_avail_res = self.avail_slot[0 : job.len, :] - job.res_vec
        if np.all(new_avail_res[:] >= 0):
            allocated = True
            self.avail_slot[0 : job.len] = new_avail_res
            job.start(self.current_time)
            self.running_job.append(job)
            assert job.start_time != -1
            assert job.finish_time != -1
            assert job.finish_time > job.start_time
        return allocated

    def allocate_job_async(self, job=Job.Job()):
        """
        Allocate the Job to this Machine
        async allocate job, not used
        """
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i : i + job.len, :] - job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True

                self.avail_slot[i : i + job.len] = new_avail_res
                job.start(self.current_time + i)

                self.running_job.append(job)

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                break
        return allocated

    def step(self):
        """
        process time
        """
        self.reward = -self.price

        self.avail_slot[:-1, :] = self.avail_slot[1:, :]
        self.avail_slot[-1, :] = self.res_slot

        # 检查任务是否结束. 取得收益
        for job in self.running_job:
            if job.start_time <= self.current_time:
                self.reward += job.pay
            if job.finish_time <= self.current_time:
                self.reward += job.pay
                self.finished_job.append(job)
                self.finished_job_num += 1
                self.running_job.remove(job)

        self.current_time += 1
        self.earning += self.reward
        self.request_job = None
        return self.reward

    def show_running_job(self):
        print("Running Jobs:")
        print([job.id for job in self.running_job])

    def show_available_slot(self):
        print("Machine #", self.id)
        print("Resource slots:")
        print(self.res_slot)
        print("Available slots:")
        self.slot_show.compute_chart(self.avail_slot)

    def static_info(self):
        return {
            "id": self.id,
            "res_slot": self.res_slot,
            "cost_vector": self.cost_vector,
        }

    def slots(self):
        return SlotShow(self.res_slot, self.avail_slot).compute_dict(self.avail_slot)

    def info(self):
        return {
            "id": self.id,
            "reward": self.reward,
            "earning": self.earning,
            "action": self.action,
            "bid": self.bid,
            "finished_job_num": self.finished_job_num,
            "running_job": self.running_job,
        }

    def show_res_vec(self):
        """
        Purpose:
        """
        res_vec = [" ".join([str(c) for c in i]) for i in self.avail_slot.T]
        print("Resources Vector")
        for i in range(len(self.res_slot)):
            print("- Resources #", i, ":", res_vec[i])

    def show(self, verbose=False):
        """

        Purpose: show the state of this machine
        """

        table = prettytable.PrettyTable(
            [
                "id",
                "Current Time",
                "Number of Res",
                "Time Horizon",
                "Resource Slot",
                "Reward",
                "Cost Vector",
                "Number of Running Jobs",
            ]
        )
        table.add_row(
            [
                self.id,
                self.current_time,
                self.num_res,
                self.time_horizon,
                self.res_slot,
                self.reward,
                self.cost_vector,
                str(len(self.running_job)),
            ]
        )
        table.title = "Machine Info"
        print(table)
        self.show_available_slot()
        self.show_res_vec()
        self.show_running_job()
        print(self.reward)

    def __str__(self) -> str:
        table = prettytable.PrettyTable(
            [
                "id",
                "Current Time",
                "Number of Res",
                "Time Horizon",
                "Resource Slot",
                "Reward",
                "Cost Vector",
                "Number of Running Jobs",
            ]
        )
        table.add_row(
            [
                self.id,
                self.current_time,
                self.num_res,
                self.time_horizon,
                self.res_slot,
                self.reward,
                self.cost_vector,
                str(len(self.running_job)),
            ]
        )
        table.title = "Machine Info"
        print(table)
        self.show_available_slot()
        self.show_res_vec()
        self.show_running_job()
        print(self.reward)
        return table.get_string() + "\n" + str(self.reward)


class Cluster:
    def __init__(
        self,
        job_backlog_size=10,
        job_slot_size=10,
        num_res=2,
        time_horizon=20,
        current_time=0,
    ):
        self.number = 0
        self.job_backlog_size = job_backlog_size
        self.job_slot_size = job_slot_size
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time
        self.now_id = 0
        self.machines = []

    def add_machine(self, res_slot, cost_vector, price):
        self.machines.append(
            Machine(
                self.now_id,
                self.num_res,
                self.time_horizon,
                self.job_slot_size,
                self.job_backlog_size,
                res_slot,
                cost_vector,
                self.current_time,
                price=price,
            )
        )
        self.now_id += 1

    def add_typed_machine(self, machine_type=MachineType(1, 2, 1), cost_vector=[2, 4]):
        res_slot = [machine_type.vCPUs, machine_type.memory]
        price = machine_type.price
        self.machines.append(
            Machine(
                self.now_id,
                self.num_res,
                self.time_horizon,
                self.job_slot_size,
                self.job_backlog_size,
                res_slot,
                cost_vector,
                self.current_time,
                price=price,
            )
        )
        self.now_id += 1

    def generate_machines_random(
        self,
        num,
        machine_average_res_vec=[20, 40],
        machine_average_cost_vec=[2, 4],
        bias_r=5,
    ):
        """
        Purpose: Randomly generate machines
        """
        for i in range(num):
            bias_r = np.random.randint(-bias_r, bias_r, self.num_res)
            res_slot = machine_average_res_vec + bias_r
            price = np.dot(machine_average_cost_vec, res_slot)
            self.add_machine(
                res_slot=res_slot, cost_vector=machine_average_cost_vec, price=price
            )
        self.number = len(self.machines)

    def generate_machines_fixed(
        self,
        groups=12,
        machine_types={
            "small": MachineType(1, 2, 1),
            "middle": MachineType(2, 4, 2),
            "big": MachineType(4, 8, 4),
        },
        machine_average_cost_vec=[2, 4],
    ):
        """
        Purpose: Fixed generate machines
        """
        for i in range(groups):
            self.add_typed_machine(machine_types["small"], machine_average_cost_vec)
            self.add_typed_machine(machine_types["big"], machine_average_cost_vec)
            self.add_typed_machine(machine_types["middle"], machine_average_cost_vec)

        self.number = len(self.machines)

    def allocate_job(self, machine_id=0, job=Job.Job()):
        self.machines[machine_id].allocate_job(job)

    def step(self):
        reward = [machine.step() for machine in self.machines]
        self.current_time += 1
        return reward

    def get_finish_job_total(self):
        return sum([machine.finished_job_num for machine in self.machines])

    def observe(self):
        return [machine.observe() for machine in self.machines]

    def clear_job(self):
        for machine in self.machines:
            machine.clear_job()

    def reset(self):
        self.current_time = 0
        for machine in self.machines:
            machine.reset()

    def get_machine(self, machine_id):
        return self.machines[machine_id]

    def show(self):
        table = prettytable.PrettyTable(
            ["id", "Resource Slot", "Reward", "Cost Vector", "Running Job"]
        )
        for machine in self.machines:
            table.add_row(
                [
                    machine.id,
                    machine.res_slot,
                    machine.reward,
                    machine.cost_vector,
                    [job.id for job in machine.running_job],
                ]
            )
            machine.show_available_slot()
        print(table)


class Bids:
    def __init__(self, cluster=Cluster(), job=Job.Job()):
        self.machines = [cluster.get_machine(i) for i in job.restrict_machines]
        self.job = job
        self.can_allocate = []
        self.bids = []
        self.request_bids()

    def get_bids(self):
        self.bids = [machine.get_bid() for machine in self.can_allocate]

    def request_bids(self):
        for machine in self.machines:
            if machine.can_allocate(self.job):
                machine.request_auction(self.job)
                self.can_allocate.append(machine)

    def __str__(self):
        table = prettytable.PrettyTable(
            ["Machine " + str(machine.id) for machine in self.can_allocate]
        )
        table.add_row([f"{bid:.2f}" for bid in self.bids])
        return table.get_string()


class JobCollection:
    def __init__(self, collection=[Job.Job()]) -> None:
        self.collection = collection


# 将JobCollection的迭代器传入MachineRestrict的迭代器,返回一个迭代器


class MachineRestrict:
    def __init__(
        self,
        cluster=Cluster(),
        collection=Job.JobCollection(),
        max_machines=10,
        min_machines=3,
    ) -> None:
        self.cluster = cluster
        self.collections = collection
        self.iter_collection = iter(collection)
        self.collection = []
        self.max_machines = max_machines
        self.min_machines = min_machines

    def reset(self):
        self.collections.reset()
        self.iter_collection = iter(self.collections)
        self.collection = []

    def generate_restrict(self):
        # 生成限制机器,首先随机生成一个机器的下标,然后随机生成一个机器的数量,打乱机器的顺序,取前面的机器
        #
        collection = next(self.iter_collection)
        for jobs in collection:
            for job in jobs:
                min_machine_num = np.random.randint(
                    0, self.cluster.number - self.max_machines
                )
                t = np.random.randint(self.min_machines, self.max_machines)
                array = np.arange(
                    min_machine_num, min_machine_num + self.max_machines + 1
                )
                np.random.shuffle(array)
                # TODO 限制机器的数量,验证是否有空槽位
                job.restrict_machines = array[:t]
        self.collection = collection
        return collection

    def show(self):
        table = prettytable.PrettyTable(["Job Id", "Enter Time", "Restrict Machine"])
        for jobs in self.collection:
            for job in jobs:
                table.add_row([job.id, job.enter_time, job.restrict_machines])
        print(table)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_restrict()


class ListIterator:
    def __init__(self, iterator):
        self.iterator = iterator  # 输入的迭代器
        self.list = []  # 存储列表的变量
        self.index = 0  # 当前列表的索引

    def __iter__(self):
        return self  # 返回自身作为迭代器

    def __next__(self):
        if self.index >= len(self.list):  # 如果索引超出了列表的长度
            self.list = next(self.iterator)  # 调用输入的迭代器产生一个新的列表
            self.index = 0  # 重置索引
        element = self.list[self.index]  # 获取当前列表的元素
        self.index += 1  # 索引加一
        return element  # 返回元素


# 定义一个可迭代的类
class NestedList:
    # 初始化方法，接受一个列表的列表作为参数
    def __init__(self, nested_list):
        # 把参数赋值给实例属性
        self.nested_list = iter(nested_list)
        # 初始化当前的子列表和索引
        self.sublist = next(self.nested_list)
        self.index = 0

    # 定义一个__iter__方法，返回一个迭代器对象
    def __iter__(self):
        # 返回自身作为迭代器对象
        return self

    # 定义一个__next__方法，返回下一个元素
    def __next__(self):
        # 如果当前的索引小于子列表的长度，就返回子列表中的元素，并增加索引

        if self.index < len(self.sublist):
            value = self.sublist[self.index]
            self.index += 1
            return value
        # 如果当前的索引等于子列表的长度，就抛出一个StopIteration异常，并清空子列表，以便下次迭代时获取下一个子列表
        else:
            self.index = 0
            self.sublist = next(self.nested_list)

            raise StopIteration


class Quote:
    def __init__(self, job=Job.Job(), cluster=Cluster()) -> None:
        self.job = job
        self.cluster = cluster.machines
        self.quotes = []
        self.get_price_set()
        pass

    def get_price_set(self):
        for machine_id in self.job.restrict_machines:
            self.quotes.append((self.cluster[machine_id].get_price(self.job)))
        return self.quotes

    def get_pay(self):
        arr = np.array(self.quotes)
        min_value = np.min(arr)
        min_index = np.argmin(arr)
        self.job.pay = min_value
        self.job.running_machine = self.job.restrict_machines[min_index]

    def show(self):
        table = prettytable.PrettyTable(["Machine", "Price"])
        for item in self.quotes:
            table.add_row([*item])

        print(table)






class AllocationMechanism:
    """Base class for allocation mechanisms"""

    def __init__(self):
        pass

    def allocate(self, bids):
        winners = np.argsort(bids.bids)[0]
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        second_prices = sorted_bids[1]
        return bids.machines[winners], prices, second_prices


class FirstPrice(AllocationMechanism):
    """(Generalised) First-Price Allocation"""

    def __init__(self):
        super().__init__()

    def allocate(self, bids):
        winners = int(np.argsort(bids.bids)[0])
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        if len(sorted_bids) == 1:
            second_prices = prices
        else:
            second_prices = sorted_bids[1]
        return bids.can_allocate[winners], prices, second_prices

    def find_second_price(self, bids):
        winners = np.argmin(bids.bids)
        prices = bids.bids[winners]
        second_prices = np.partition(bids.bids, 1)[1]
        return bids.machines[winners], prices, second_prices


class SecondPrice(AllocationMechanism):
    """(Generalised) Second-Price Allocation"""

    def __init__(self):
        super().__init__()

    def allocate(self, bids):
        winners = np.argsort(bids.bids)[0]
        prices = np.sort(bids.bids)[1]
        return bids.can_allocate[winners], prices, prices






class ReverseAuction:
    def __init__(
        self,
        cluster=Machine.Cluster(),
        allocation_mechanism=AllocationMechanism.AllocationMechanism(),
    ) -> None:
        self.cluster = cluster
        self.allocation_mechanism = allocation_mechanism

    def auction(self, job=Job.Job()):
        self.current_job = job
        self.bids = Machine.Bids(self.cluster, self.current_job)

        self.bids.get_bids()
        bids = self.bids
        if len(bids.can_allocate) == 0:
            bids.job.pay = 0
            bids.job.running_machine = -1
            return False
        winner_machine, prices, _ = self.allocation_mechanism.allocate(bids)
        if prices > bids.job.budget:
            bids.job.pay = 0
            bids.job.running_machine = -1
            # print("Auction Failed: " + str(bids.job))
            return False
        bids.job.pay = prices
        bids.job.running_machine = winner_machine.id
        winner_machine.allocate_job(bids.job)
        # print([mac.id for mac in bids.machines])
        # print(bids)
        # print(
        #    "Auction Success: "
        #    + "Job:"
        #    + str(bids.job.id)
        #    + " Pay:"
        #    + str(prices)
        #    + " Machine:"
        #    + str(winner_machine.id)
        # )
        return True







class VehicleJobSchedulingParameters:
    def __init__(
        self,
        distribution="bi_model_dist",
        average_per_slot=10,
        duration=30,
        machine_numbers=12,
        random_cluster=False,
    ):
        # Job Config
        self.max_job_vec = [24, 100]
        self.max_job_len = 20
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
        self.allocation_mechanism = AllocationMechanism.FirstPrice()
        self.auction_type = Auction.ReverseAuction(
            self.cluster, self.allocation_mechanism
        )
        # observation, action space
        self.action_space_continuous = False
        self.action_discrete_space = 20
        self.action_space_low = 1 / 2
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
        self.__agent_selector = self._agent_selector()
        self.agent_selection = self.__agent_selector.next()
        self.round_start = True
        self.round_jobs = None
        self.pay = [0 for _ in range(len(self.agents))]
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
        # print("Current time:", self.parameters.cluster.current_time)
        self.finished_job = self.parameters.cluster.get_finish_job_total()
        self.parameters.finished_job = self.finished_job
        self.parameters.total_job = self.total_job
        self.parameters.time_step = self.parameters.cluster.current_time
        if self.parameters.stop_condition():
            self.truncations = {agent: True for agent in self.agents}
            self.terminations = {agent: True for agent in self.agents}
            self.done = True
            # print("Finished!!!")
        self.parameters.cluster.clear_job()

    # @functools.lru_cache(maxsize=1000)
    def action(self, action):
        if self.parameters.action_space_continuous:
            return float(action[0])
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
