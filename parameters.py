import math
import numpy as np

import job_distribution



'''
定义相关参数
'''
class Parameters:
    def __init__(self):

        # NOTE - 训练相关参数
        self.output_filename = 'data/tmp'

        self.num_epochs = 10000
        self.simulate_len = 10
        self.num_ex = 1

        self.output_frequent = 10

        self.num_seq_per_batch = 10
        self.episode_max_length = 200

        # NOTE - 服务器配置相关
        self.num_res = 2
        self.num_nw = 5

        self.time_horizon = 20
        self.max_job_len = 15
        self.res_slot = 15
        self.max_job_size = 10

        # NOTE - 日志相关
        self.backlog_size = 60
        # 从最新的任务开始追踪多少个时间步
        self.max_track_since_new = 10
        # 当前工作最大不同颜色数
        self.job_num_cap = 40

        # NOTE - 任务相关
        # 泊松分布的lambda值
        self.new_job_rate = 0.7

        # 折损系数
        self.discount = 1

        # 任务分布
        self.dist = job_distribution.Dist(self.num_res,
                                         self.max_job_size, self.max_job_len)

        # NOTE - 图形表示
        # 确信可以被图形表示
        assert self.backlog_size % self.time_horizon == 0
        self.backlog_width = int(math.ceil(self.backlog_size
                                           / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = (self.res_slot +
                                    self.max_job_size * self.num_nw * self.num_res +
                                    self.backlog_width+1)

        # NOTE - 压缩表示

        # +1是积压指示
        self.network_compact_dim = (self.num_res+1) *\
            (self.time_horizon+self.num_nw)+1
        # +1是空(Void)动作
        self.network_output_dim = self.num_nw + 1
        # 工作视野中延迟的惩罚项
        self.delay_penalty = -1
        # 持有当前视野中的惩罚项
        self.hold_penalty = -1
        # 由于队列满导致错过而产生的惩罚
        self.dismiss_penalty = -1
        # 需要组合和处理的帧数
        self.num_frame = 1

        # NOTE - 学习率相关
        self.lr_rate = 0.001
        # RMS Prop
        self.rms_rho = 0.9
        self.rms_eps = 1e-9

        # NOTE - 更改随机数种子测试未预见的情况
        self.unseen = False

        # NOTE - 监督学习模仿的策略
        self.batch_size = 10
        self.evaluate_policy_name = 'SJF'

    # NOTE - 计算相关的参数
    def compute_dependent_parameters(self):
        """
        Purpose: 
        """
        assert self.backlog_size % self.time_horizon == 0
        self.backlog_width = self.backlog_size/self.time_horizon
        self.network_input_height = self.time_horizon
        # +1代表从最新任务开始的时间信息
        self.network_input_width = (self.res_slot +
                                    self.max_job_size*self.num_nw) * self.num_res +\
            self.backlog_size+1

        # NOTE - 压缩表示
        # +1表示积压指示
        self.network_compact_dim = (self.num_res+1) *\
            (self.time_horizon+self.num_nw)+1

        # +1表示空动作
        self.network_output_dim = self.num_nw + 1
