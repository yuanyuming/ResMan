import numpy as np
import prettytable


class Machine:
    def __init__(self, num_res, time_horizon, res_slot, job_num_cap) -> None:
        '''
        Initializes the machine
        '''

        self.num_res = num_res
        self.time_horizon = time_horizon
        self.res_slot = res_slot

        self.avail_slot = np.ones((self.time_horizon, self.num_res))\
            * self.res_slot
        self.running_job = []

    def allocate_job(self, job, curr_time):
        '''
            Allocate the Job to this Machine
        '''
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i:i+job.len, :]-job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True

                self.avail_slot[i:i+job.len] = new_avail_res
                job.start_time = curr_time + i
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avail_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        print(avail_slot[:job.res_vec[res]])

                break
            return allocated

    def time_proceed(self, curr_time):
        '''
        process time
        '''

        self.avail_slot[:-1, :] = self.avail_slot[1:, :]
        self.avail_slot[-1, :] = self.res_slot

        for job in self.running_job:
            if job.finish_time <= curr_time:
                self.running_job.remove(job)

    def show(self):
        """
        show the state of this machine
        Purpose: one
        """

        table = prettytable.PrettyTable(
            ["Number of Res", "Time Horizon", "Number of Running Jobs"])
        table.add_row([self.num_res, self.time_horizon, len(self.running_job)])
        table.set_style(prettytable.MSWORD_FRIENDLY)
        table.title = "Machine Info"
        print(table)
        print("Resource slots:")
        print(self.res_slot)
        print("Available slots:")
        print(self.avail_slot)
        print("Running Jobs:")
        print(self.running_job)
