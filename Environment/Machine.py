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
            new_avail_res = self.avail_slot[i:i+job.len,:]-job.res_vec
            if np.all(new_avail_res[:]>=0):
                allocated = True
                
                self.avail_slot[i:i+job.len] =new_avail_res
                job.start_time = curr_time +i
                job.finish_time = job.start_time + job.len
                
                self.running_job.append(job)
                
                # 图形表示
                used_color = np.unique(self.canvas[:])
                
                # 应有足够的颜色
                for color in self.colormap:
                    if color not in self.colormap:
                        new_color = color
                        break
                    
                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.finish_time
                
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                
                for res in range(self.num_res):
                    for i in range(canvas_start_time,canvas_end_time):
                        avail_slot = np.where(self.canvas[res,i,:]==0)[0]
                        self.canvas[res,i,avail_slot[:job.res_vec[res]]] = new_color
                        
                break
            return allocated
        
        def time_proceed(self,curr_time):
            
            self.avail_slot[:-1,:] = self.avail_slot[1:,:]