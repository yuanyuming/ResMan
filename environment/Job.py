"""
定义任务Job相关结构
"""
from collections import OrderedDict

# 导入numpy库
import numpy as np
import prettytable
from gymnasium import spaces
from scipy.stats import poisson

# 定义一个类，表示任务分配的参数


class JobDistribution:
    # 初始化方法，接受以下参数：
    # max_job_vec: 一个列表，表示每种资源的最大任务数量
    # max_job_len: 一个整数，表示任务的最大时长
    # job_small_chance: 一个浮点数，表示任务是小任务的概率
    # job_priority_range: 一个列表，表示任务的优先级范围
    def __init__(
        self,
        max_job_vec=[10, 20],
        max_job_len=10,
        job_small_chance=0.8,
        job_priority_range=[1, 5],
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
    def normal_dist(self):
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
        res_vec=[2, 3],
        job_len=5,
        job_id=0,
        enter_time=0,
        priority=1,
        average_cost_vec=[4, 6],
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
        self.job_vec = self.generate_job()  # 生成的任务向量
        self.average_cost_vec = average_cost_vec  # 平均成本向量
        self.budget = self.calculate_budget(average_cost_vec)  # 计算的预算
        self.pay = 0  # 支付金额
        self.utility = 0  # 效用值

    # 定义一个方法，根据给定的分布函数随机生成任务时长和资源需求向量，并更新预算和任务向量
    def random_job(self, dist=JobDistribution().bi_model_dist):
        self.len, self.res_vec = dist()  # 调用分布函数生成时长和需求向量
        self.budget = self.calculate_budget(self.average_cost_vec)  # 更新预算
        self.job_vec = self.generate_job()  # 更新任务向量

    # 定义一个方法，根据资源需求向量生成任务向量，即每个时刻需要的资源数量
    def generate_job(self):
        return [self.res_vec] * self.len

    # 定义一个方法，根据每种资源的平均成本和方差计算任务的预算，并保证预算不为负数
    def calculate_budget(self, average_cost_vec, var=0.3):
        return max(
            0,
            (np.dot(self.res_vec, average_cost_vec) + var * np.random.normal())
            * self.len
            * self.priority,
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
        print(self.job_vec)

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
                        job.res_vec,
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
                        job.res_vec,
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
                        job.res_vec,
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
