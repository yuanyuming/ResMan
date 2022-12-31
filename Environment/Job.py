import Parameters
'''
定义任务Job相关结构
'''


class Job:
    '''
    res_vec:资源需求向量
    job_len:任务长度
    job_id:任务id,唯一
    enter_time:进入队列的时间 
    #TODO - 具体安排是否需要进入队列的时间,因为考虑到多机分配问题 
    '''
    def __init__(self,res_vec,job_len,job_id,enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        
class JobSlot:
    def __init__(self,pa = Parameters):
        self.slot = pa
        
        