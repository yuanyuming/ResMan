import numpy as np
import math
import gym
import Job
import Machine
import matplotlib.pyplot as plt
import parameters

class Allocation_Environment :
    def __init__(self,pa=parameters.Parameters() ,nw_len_seq=None,nw_size_seq=None,
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
        self.machine = Machine.Machine(pa.num_res, pa.time_horizon, pa.res_slot, pa.job_num_cap)
        self.job_slot = Job.JobSlot(pa.num_nw)
        self.job_backlog = Job.JobBacklog(pa.backlog_size)
        self.job_record = Job.JobRecord()
        self.extra_info = ExtraInfo(pa.max_track_since_new)
        
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
    def get_new_job_from_seq(self, seq_num,seq_idx):
        """
        Purpose: self
        """
        new_job = Job.Job(res_vec=self.nw_size_seq[seq_num,seq_idx,:],
                          job_len=self.nw_len_seq[seq_num,seq_idx],
                          job_id=len(self.job_record.record),
                          enter_time=self.curr_time)
        return new_job
        
    def image_repre(self):
        """
        Purpose: self
        """
        backlog_width = int(math.ceil(self.pa.backlog_size
            / float(self.pa.time_horizon)))
        image_repr = np.zeros((self.pa.network_input_height,\
            self.pa.network_input_width))
        
        ir_pt = 0
        for i in range(self.pa.num_res):
            image_repr[:,ir_pt,ir_pt+self.pa.res_slot] = \
                self.machine.canvas[i, :, :]
                
            for j in range(self.pa.num_nw):
                if (self.job_slot[j] is not None ):
                    image_repr[: self.job_slot.slot[j].len,
                                ir_pt: ir_pt + 
                                self.job_slot.slot[j].res_vec[i]] = 1
                    # comment: 
                # end if
                ir_pt += self.pa.max_job_size
                
        image_repr[:self.job_backlog.curr_size/backlog_width,
                    ir_pt:ir_pt+
                    self.job_backlog.curr_size%backlog_width] =1
        ir_pt += backlog_width
        
        image_repr[:,ir_pt:ir_pt+1]=self.extra_info.time_since_last_job/\
            float(self.extra_info.max_tracking_time_since_last_job)
        return image_repr
        
    # end def
    def compact_repre(self):
        """
        Purpose: self
        """
        compact_repre = np.zeros(self.pa.time_horizon*(self.pa.num_res+1)+ \
        self.pa.num_nw*(self.pa.num_res+1)+1,dtype=float)
        cr_pt=0
        # 当前任务奖励,每一步后,机器中存在多少任务
        job_allcoated = np.ones(self.pa.time_horizon) * len(self.machine.running_job)
        
        for job in self.machine.running_job:
            job_allcoated[job.finish_time - self.curr_time:]-=1
        compact_repre[cr_pt:cr_pt+self.pa.time_horizon] = job_allcoated
        cr_pt+= self.pa.time_horizon
        
        # 当前工作允许的时间间隙
        for i in range(self.pa.num_res):
            compact_repre[cr_pt:cr_pt+self.pa.time_horizon] = self.machine.avail_slot[:,i]
            
        # 新任务时长和大小
        for i in range(self.pa.num_res):
            if self.job_slot.slot[i] is None:
                compact_repre[cr_pt:cr_pt+self.pa.num_res+1] = 0
                cr_pt += self.pa.num_res +1
            else:
                compact_repre[cr_pt]+=1
                
                for j in range(self.pa.num_res):
                    compact_repre[cr_pt] = self.job_slot.slot[i].res_slot[j]
                    cr_pt+=1
                    # comment: 
                # end for
            # comment: 
        # end for
        
        # backlog queue
        compact_repre[cr_pt] = self.job_backlog.curr_size
        cr_pt += 1
        
        assert cr_pt == len(compact_repre)
        
        return compact_repre
        
    # end def
    def observe(self):
        if self.repre=="image":
            return self.image_repre()
        elif self.repre == 'compact':
            return self.compact_repre()
    def plot_state(self):
        plt.figure("screen",figsize=(20,5))
        
        skip_row = 0
        
        for i in range(self.pa.num_res):
            plt.subplot(self.pa.num_res,
                        1+self.pa.new_nw+1, # 第一个+1当前任务, 最后一个+1记录队列
                        1*(self.pa.new_nw+1)+skip_row+1) # 将记录放在最后, +1以避免0
            plt.imshow(self.machine.canvas[i,:,:],interpolation='nearest',vmax=1)
            for j in range(self.pa.num_nw):
                job_slot = np.zeros((self.pa.time_horizon,self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:
                    job_slot[:self.job_slot.slot[j].len,:self.job_slot[j].res_vec[i]] = 1
                    
                plt.subplot(self.pa.num_res,
                            1+self.pa.num_nw+1,
                            1+i*(self.pa.num_nw+1)+j+skip_row+1)
                plt.imshow(job_slot,interpolation="nearest",vmax=1)
                
                if j == self.pa.num_nw -1:
                    skip_row+=1
                    
        skip_row -= 1
        backlog_width = int(math.ceil(self.pa.backlog_size/float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))
        
        backlog[:self.job_backlog.curr_size/backlog_width, backlog_width] = 1
        backlog_width[self.job_backlog.curr_size/backlog_width, : self.job_backlog.curr_size%backlog_width] =1
        
        plt.subplot(self.pa.num_res,
                    1+self.pa.num_nw+1,
                    self.pa.num_res * (self.pa.num_nw+1)+skip_row+1)
        
        extra_info = np.ones((self.pa.time_horizon,1))*\
            self.extra_info.time_since_last_job/\
                float(self.extra_info.max_tracking_time_since_last_job)
                
        plt.imshow(extra_info,interpolation='nearest',vmax=1)
        
        plt.show()
        pass
    def get_reward(self):
        """
        Purpose: 计算reward
        """
        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)
            
        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)
                
        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)
                
        return reward
        
    # end def
    # 重置环境
    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0
        
        # 初始化系统
        self.machine = Machine(self.pa)
        self.job_slot = Job.JobSlot(self.pa)
        self.job_backlog = Job.JobBacklog(self.pa)
        self.job_record = Job.JobRecord()
        self.extra_info = ExtraInfo(self.pa)
        pass
    # 执行一步
    def step(self,a,repeat=False):
        status = None
        
        done = False
        reward = 0
        info = None
        
        if a == self.pa.num_nw:
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a],self.curr_time)
            if not allocated:
                status = 'MoveOn'
            else:
                status = 'Allocate'
                
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()
            
            # 添加新任务
            self.seq_idx += 1
            
            # 判断是否结束
            if self.end == 'no_new_job':
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == 'all_done':
                if self.seq_idx >= self.pa.simu_len and \
                    len(self.machine.running_job) ==0 and \
                    all(s is None for s in self.job_slot.slot) and \
                    all(s is None for s in self.job_backlog.backlog):
                        done = True
                elif self.curr_time > self.pa.episode_max_length:
                    done = True
                    
            if not done:
                if self.seq_idx < self.pa.simu_len:
                    new_job = self.get_new_job_from_seq(self.seq_no,self.seq_idx)
                    
                    if new_job.len > 0:
                        to_backlog = True
                        
                        for i in range(self.pa.nun_nw):
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break
                            
                            if to_backlog:
                                if self.job_backlog.curr_size < self.pa.backlog_size:
                                    self.job_backlog.backlog[self.job_backlog.curr_size]= new_job
                                    self.job_backlog.curr_size += 1
                                    self.job_record.record[new_job.id] = new_job
                                else:
                                    print("Backlog is FULL!")
                                    
            reward = self.get_reward()
            
        elif status == 'Allocate':
            self.job_record.record[self.job_slot[a].id] = self.job_slot.slot[a]
            self.job_slot.slot[a] = None
            
            # backlog 退队
            if self.job_backlog.curr_size > 0 :
                self.job_slot.slot[a] = self.job_backlog.backlog[0]
                self.job_backlog.backlog[:-1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -=1
                
        ob = self.observe()
        
        info = self.job_record
        
        if done:
            self.seq_idx = 0
            
            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
                
            self.reset()
            
        if self.render:
            self.plot_state()
            
        return ob, reward, done, info
        pass
    def close(self):
        pass


class ExtraInfo:
    def __init__(self,max_track_time_since_new) -> None:
        self.time_since_last_job = 0
        self.max_tracking_time_since_last_job = max_track_time_since_new
        
        def new_job_comes(self):
            self.time_since_last_job = 0
            
        def time_proceed(self):
            if self.time_since_last_job < self.max_tracking_time_since_last_job:
                self.time_since_last_job += 1

