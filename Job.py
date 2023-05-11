"""
定义任务Job相关结构
"""
from matplotlib import collections
import prettytable
import numpy as np

from scipy.stats import poisson
from sympy import false, true


"""
Parameters:
JobDistribution: 
- max new work vector
- max job len

JobCollection:
- averge jobs per slot
- id start
- enter time
- duration
- Dist

"""


class JobDistribution:
    def __init__(
        self,
        max_job_vec=[10, 20],
        max_job_len=10,
        job_small_chance=0.8,
        job_priority_range=[1, 5],
    ):
        self.num_res = len(max_job_vec)
        self.max_nw_size = max_job_vec
        self.max_job_len = max_job_len

        self.job_small_chance = job_small_chance

        self.job_len_big_lower = int(max_job_len * 2 / 3)
        self.job_len_big_upper = max_job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = int(max_job_len / 5)

        self.dominant_res_lower = np.divide(np.array(max_job_vec), 2)
        self.dominant_res_upper = max_job_vec

        self.other_res_lower = 1
        self.other_res_upper = np.divide(np.array(max_job_vec), 5)
        self.priority_range = job_priority_range

    def __str__(self) -> str:
        return f"JobDistribution({self.max_nw_size}, {self.max_job_len}, {self.job_small_chance})"

    def priority_dist(self):
        return np.random.randint(self.priority_range[0], self.priority_range[1] + 1)

    def normal_dist(self):
        # NOTE - 新任务时长
        nw_len = np.random.randint(1, self.max_job_len + 1)

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            # comment:
            nw_size[i] = np.random.randint(1, self.max_nw_size[i] + 1)
        # end for
        return nw_len, nw_size

    def bi_model_dist(self):
        # NOTE - 新任务时长
        if np.random.rand() < self.job_small_chance:
            nw_len = np.random.randint(
                self.job_len_small_lower, self.job_len_small_upper + 1
            )
        else:  # 大任务
            nw_len = np.random.randint(self.job_len_big_lower, self.job_len_big_upper)

        # NOTE - 任务资源请求
        dominant_res = np.random.randint(0, self.num_res)
        nw_size = np.zeros([self.num_res])
        for i in range(self.num_res):
            # comment:
            if i == dominant_res:
                nw_size[i] = np.random.randint(
                    self.dominant_res_lower[i], self.dominant_res_upper[i] + 1
                )
            else:
                nw_size[i] = np.random.randint(
                    self.other_res_lower, self.other_res_upper[i] + 1
                )

        return nw_len, nw_size


class Job:
    """
    res_vec:资源需求向量
    job_len:任务长度
    job_id:任务id,唯一
    enter_time:进入队列的时间
    # TODO - Job 直接传参Dist
    """

    def __init__(
        self,
        res_vec=[2, 3],
        job_len=5,
        job_id=0,
        enter_time=0,
        priority=1,
        average_cost_vec=[4,6],
    ):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.restrict_machines = [1, 2]
        self.running_machine = 0
        self.enter_time = enter_time
        self.time_restrict = 0
        self.start_time = -1
        self.finish_time = -1
        self.priority = priority
        self.job_vec = self.generate_job()
        self.budget = self.calculate_budget(average_cost_vec)
        self.pay = 0
        self.utility = 0
        self.running_machine = -1

    def random_job(self, dist=JobDistribution().bi_model_dist):
        self.len, self.res_vec = dist()
        self.job_vec = self.generate_job()

    def generate_job(self):
        """
        Purpose:
        """
        return [self.res_vec] * self.len
    def calculate_budget(self, average_cost_vec, var=0.3):
        """
        Purpose:
        """
        return (np.dot(self.res_vec, average_cost_vec)+var*np.random.normal()) * self.len*self.priority
    def read_job_from_file(self):
        """
        Purpose:
        """

    def request(self):
        return self.id, self.res_vec, self.len, self.priority, self.restrict_machines

    def start(self, curr_time):
        """
        Purpose:
        """
        self.start_time = curr_time
        self.finish(curr_time + self.len)

    def finish(self, finish_time):
        self.finish_time = finish_time

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

    def show(self):
        """
        Purpose: show the job
        """
        table = prettytable.PrettyTable(
            [
                "Job Id",
                "Res Vector",
                "Job Len",
                "Priority",
                "Budget"
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

    def get_pay(self, pay=0):
        self.pay = pay

    def __str__(self):
        return "id:{},Res Vector:{},Job Len:{},Restrict Machine:{}".format(
            self.id, self.res_vec, self.len, self.restrict_machines
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
        averge_cost_vec = [4,6]
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
            job = Job(job_res_vec, job_len, id, self.enter_time, self.priority(), self.average_cost_vec)
            job.enter_time = self.enter_time
            job.id = id
            job.priority = self.priority()
            job.random_job(self.Dist)
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
                job = Job()
                job.enter_time = self.enter_time
                job.id = id
                job.random_job(self.Dist)
                job.priority = self.priority()
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
