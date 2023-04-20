"""
定义任务Job相关结构
"""
from matplotlib import collections
import prettytable
import numpy as np

from scipy.stats import poisson


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
    def __init__(self, max_job_vec=[10, 20], max_job_len=10, job_small_chance=0.8):
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

    def __str__(self) -> str:
        return f"JobDistribution({self.max_nw_size}, {self.max_job_len}, {self.job_small_chance})"

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
    #TODO - 具体安排是否需要进入队列的时间,因为考虑到多机分配问题
    """

    def __init__(self, res_vec=[2, 3], job_len=5, job_id=0, enter_time=0):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.restrict_machines = []
        self.running_machine = 0
        self.enter_time = enter_time
        self.time_restrict = -1
        self.start_time = -1
        self.finish_time = -1
        self.job_vec = self.generate_job()
        self.pay = 0

    def random_job(self, dist=JobDistribution().bi_model_dist):
        self.len, self.res_vec = dist()
        self.job_vec = self.generate_job()

    def generate_job(self):
        """
        Purpose:
        """
        return [self.res_vec] * self.len

    def read_job_from_file(self):
        """
        Purpose:
        """

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
            self.restrict_machines,
            self.enter_time,
            self.start_time,
            self.finish_time,
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
                "Restrict Machines",
                "Enter Time",
                "Start Time",
                "Finish Time",
            ]
        )
        table.add_row(self.to_list())
        table.set_style(prettytable.MSWORD_FRIENDLY)
        table.title = "Job Info"
        print(table)
        print("Job Vector")
        print(self.job_vec)

    def get_pay(self, price_set=[5, 7]):
        pass

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
    ):
        self.average = average
        self.id_start = id_start
        self.enter_time = enter_time
        self.Dist = job_dist
        self.now_id = id_start
        self.duration = duration

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
            job = Job()
            job.enter_time = self.enter_time
            job.id = id
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

    def add_new_job(self, job):
        """
        Purpose:
        """
        for i in range(len(self.slot)):
            if self.slot[i] is None:
                self.slot[i] = job
                break

    def select_job(self, num=0):
        job = self.slot[num]
        self.slot[num] = None
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
        return str(table)


class JobBacklog:
    """
    Backlog the jobs
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

    def add_backlog(self, job):
        """
        Purpose:
        """
        if self.curr_size < self.backlog_size:
            self.backlog[self.curr_size] = job
            self.curr_size += 1
        else:
            self.backlog[:-1] = self.backlog[1:]
            self.backlog[-1] = job

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
        return str(table)


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
