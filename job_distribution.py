import numpy as np
import Parameters

'''
任务生成
'''


class Dist:
    def __init__(self,num_res,max_nw_size,job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len
        
        self.job_small_chance = 0.8
        
        self.job_len_big_lower = job_len * 2/3
        self.job_len_big_upper = job_len
        
        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len/5
        
        self.dominant_res_lower = max_nw_size /2
        self.dominant_res_upper = max_nw_size
        
        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5
        
    def normal_dist(self):
        
        #NOTE - 新任务时长
        nw_len = np.random.randint(1, self.job_len+1)
        
        nw_size = np.zeros(self.num_res)
        
        for i in range(self.num_res):
            # comment: 
            nw_size[i] = np.random.randint(1, self.max_nw_size+1)
        # end for
        return nw_len,nw_size
    
    def bi_model_dist(self):
        
        #NOTE - 新任务时长
        if np.random.rand() < self.job_small_chance:
            nw_len = np.random.randint(self.job_len_small_lower,\
                self.job_len_small_upper+1)
        else:#大任务
            nw_len = np.random.randint(self.job_len_big_lower,\
                self.job_len_big_upper)
        
        #NOTE - 任务资源请求
        dominant_res = np.random.randint(0,self.num_res) 
        nw_size = np.zeros([self.num_res])
        for i in range(self.num_res):
            # comment: 
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,\
                    self.dominant_res_upper+1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,\
                    self.other_res_upper+1)
        
        return nw_len, nw_size

def generate_sequence_work(pa=Parameters.Parameters(),seed=29):
    np.random.seed(seed)
    simulate_len = pa.simulate_len * pa.num_ex
    
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = np.zeros(simulate_len,dtype=np.int64)
    nw_size_seq = np.zeros((simulate_len,pa.num_res),dtype=np.int8)
    
    for i in range(simulate_len):
        # comment: 
        if np.random.rand()<pa.new_job_rate:
            nw_len_seq[i],nw_size_seq[i,:]=nw_dist()
            
    nw_len_seq= np.reshape(nw_len_seq,[pa.num_ex,pa.simulate_len])
    nw_size_seq = np.reshape(nw_size_seq,[pa.num_ex,pa.simulate_len,pa.num_res])
    
    return nw_len_seq,nw_size_seq
