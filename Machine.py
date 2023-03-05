import numpy as np


class Machine:
    '''
    Initializes the machine
    '''

    def __init__(self, num_res, time_horizon, res_slot, job_num_cap) -> None:
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.res_slot = res_slot

        self.avail_slot = np.ones((self.time_horizon, self.num_res))\
            * self.res_slot
        self.running_job = []

        self.colormap = np.arange(
            1/float(job_num_cap), 1, 1/float(job_num_cap))
        np.random.shuffle(self.colormap)

        self.canvas = np.zeros((num_res, time_horizon, res_slot))

    def get_color(self):
        """
        Purpose: get an unused color
        """

    # end def
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

                # 图形表示
                used_color = np.unique(self.canvas[:])

                # 应有足够的颜色
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avail_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        print(avail_slot[:job.res_vec[res]])
                        self.canvas[res, i,
                                    avail_slot[:job.res_vec[res]]] = new_color

                break
            return allocated
    '''
    process time
    '''

    def time_proceed(self, curr_time):

        self.avail_slot[:-1, :] = self.avail_slot[1:, :]
        self.avail_slot[-1, :] = self.res_slot

        for job in self.running_job:
            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # 图形表示

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

    '''
    show the state of this machine
    '''

    def show(self):
        """
        Purpose: one
        """

    # end def
