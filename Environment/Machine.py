import numpy as np
import parameters
class Machine:
    def __init__(self,pa=parameters.Parameters()) -> None:
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        
        self.avail_slot = np.ones((self.time_horizon,self.num_res))\
            *self.res_slot
        self.running_job = []
        
        self.colormap = np.arange(1/float(pa.job_num_cap,1,1/float(pa.job_num_cap)))
        np.random.shuffle(self.colormap)
        
        self.canvas = np.zeros((pa.num_res,pa.time_horizon,pa.res_slot))
        
    def allocate_job(self,job,curr_time):
        allocated = False
        
        for i in range(0,self.time_horizon - job.len):
            new_avail_res = self.avail_slot[t:t+job.len,:]-job.res_vec
            if np.all(new_avail_res[:]>=0):
                allocated = True
                
                self.avail_slot[i:i+job.len] =new_avail_res
                job.start_time = curr_time +i
                job.finish_time = job.start_time + job.len
                
                self.running_job.append(job)
                
                # 图形表示