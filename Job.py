'''
定义任务Job相关结构
'''
from msilib.schema import tables
import prettytable


class Job:
    '''
    res_vec:资源需求向量
    job_len:任务长度
    job_id:任务id,唯一
    enter_time:进入队列的时间 
    #TODO - 具体安排是否需要进入队列的时间,因为考虑到多机分配问题 
    '''

    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1

    def show(self):
        """
        Purpose: show the job
        """
        table = prettytable.PrettyTable(
            ['Job Id', 'Res Vector', 'Enter Time', 'Start Time', 'Finish Time'])
        table.add_row([self.id, self.res_vec, self.enter_time,
                      self.start_time, self.finish_time])
        table.set_style(prettytable.MSWORD_FRIENDLY)
        print(table)

    # end def


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
            ['Job Id', 'Res Vector', 'Enter Time', 'Start Time', 'Finish Time'])
        for job in self.slot:
            if job is not None:
                table.add_row([job.id, job.res_vec, job.enter_time,
                               job.start_time, job.finish_time])
        print(table)
    # end def


class JobBacklog:
    '''
        Backlog the jobs
    '''

    def __init__(self, backlog_size) -> None:
        self.backlog = [None]*backlog_size
        self.curr_size = 0

    def show(self):
        """
        Purpose: show the JobSlot
        """
        table = prettytable.PrettyTable(
            ['Job Id', 'Res Vector', 'Enter Time', 'Start Time', 'Finish Time'])
        for job in self.backlog:
            if job is not None:
                table.add_row([job.id, job.res_vec, job.enter_time,
                               job.start_time, job.finish_time])
        print(table)


class JobRecord:
    def __init__(self) -> None:
        self.record = {}
