
import numpy as np

import Parameters

def generate_sequence_work(pa=Parameters.Parameters(),seed=29):
    
    np.random.seed(seed)
    
    simulate_len = pa.simulate_len * pa.num_ex
    
    nw_dist = pa.dist.bi_model_dist
    
    nw_len_seq = np.zeros(simulate_len,dtype=int)
    nw_size_seq = np.zeros((simulate_len,pa.num_res),dtype=int)
    
    for i in range(simulate_len):
        # comment: 
        if np.random.rand()<pa.new_job_rate:
            nw_len_seq[i],nw_size_seq[i,:]=nw_dist()
            
    nw_len_seq= np.resheape(nw_len_seq,[pa.num_ex,pa.simulate_len])
    nw_size_seq = np.resheape(nw_size_seq,[pa.num_ex,pa.simulate_len,pa.num_res])
    
    return nw_len_seq,nw_size_seq