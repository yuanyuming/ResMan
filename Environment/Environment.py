import numpy as np
import gym
import Job
import Machine
import Parameters

class Allocation_Environment(gym.Enpiptfv) :
    def __init__(self,pa,nw_len_seq=None,nw_size_seq=None,
                 seed=42,render=False,repre='image',end='no_new_job') -> None:
        # 设置虚拟机
        # 创建虚拟机生成器
        self.pa = pa
        self.render = render
        self.repre = repre
        self.end = end
        
        self.nw_dist = pa.dist.bi_model_dist
        
        self.curr_time = 0
        
        if self.pa.unseen:
            np.random.seed(457869)
        else:
            np.random.seed(seed)
            
        if nw_len_seq is None or nw_size_seq is None:
            # 生成新任务
            self.nw_len_seq,self.nw_size_seq = \
                self.generate_sequence_work(self.pa.simu_len*self.pa.num_ex)
            self.workload = np.zeros(pa.num_res)
            for i in range(pa.num_res):
                self.workload[i]=\
                    np.sum(self.nw_size_seq[:,i]*self.nw_len_seq)/\
                        float(pa.res_slot)/float(len(self.nw_len_seq))
                print("Load On #",i,"Resource Dimension Is ",self.workload[i])
        else:
            self.nw_len_seq = nw_len_seq
            self.nw_size_seq = nw_size_seq
            
        self.seq_no = 0 # 队列号
        self.seq_idx = 0 # 队列中序号
        
        # 初始化系统
        self.machine = Machine.Machine(pa)
        self.job_slot = Job.JobSlot(pa)
        self.job_backlog = Job.JobBacklog(pa)
        self.job_record = Job.JobRecord()
        self.extra_info = ExtraInfo(pa)
        
        pass
    # 观察环境
    def generate_sequence_work(self,simu_len):
        nw_len_seq = np.zeros(simu_len,dtype=int)
        nw_size_seq = np.zeros((simu_len,self.pa.new_job_rate),dtype=int)
        
        for i in range(simu_len):
            if np.random.rand() < self.pa.new_job_rate:
                # 新任务到达
                nw_len_seq[i],nw_size_seq[i,:] = self.nw_dist()
                
        return nw_len_seq,nw_size_seq
    def observe(self):
        pass
    # 重置环境
    def reset(self):
        pass
    # 执行一步
    def step(self):
        pass
    def close(self):
        pass


class ExtraInfo:
    def __init__(self,pa) -> None:
        self.time_since_last_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_time_since_new
        
        def new_job_comes(self):
            self.time_since_last_job = 0
            
        def time_proceed(self):
            if self.time_since_last_job < self.max_tracking_time_since_last_job:
                self.time_since_last_job += 1

