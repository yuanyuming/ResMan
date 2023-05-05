import Job
import Machine
import gymnasium as gym


class Parameters:
    def __init__(self):
        # Job Config
        self.max_job_vec = [10, 20]
        self.max_job_len = 10
        self.job_small_chance = 0.8
        self.job_distribution = Job.JobDistribution(
            self.max_job_vec, self.max_job_len, self.job_small_chance
        )
        self.job_dist = self.job_distribution.bi_model_dist
        # Job Collection Config
        self.job_id = 0
        self.current_time = 0
        self.average = 8
        self.duration = 10
        self.max_job_vec = [10, 20]
        self.max_job_len = 10
        self.job_small_chance = 0.8
        self.job_collection = Job.JobCollection(
            self.average, self.job_id, self.current_time, self.duration, self.job_dist
        )
        # Machine Config
        self.job_backlog_size = 10
        self.job_slot_size = 10
        self.num_res = len(self.max_job_vec)
        self.time_horizon = 20
        self.current_time = 0
        self.res_slot = [20, 50]
        self.machine = Machine.Machine(
            id=0,
            num_res=self.num_res,
            time_horizon=self.time_horizon,
            current_time=self.current_time,
            job_backlog_size=self.job_backlog_size,
            job_slot_size=self.job_slot_size,
            res_slot=self.res_slot,
            cost_vector=[1, 2],
        )
        # Job Iterator
        self.job_iterator = Machine.ListIterator(iter(self.job_collection))

class SingleAgentEnvironment(gym.Env):
    
