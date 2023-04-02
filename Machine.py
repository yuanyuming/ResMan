import numpy as np
import prettytable
import Job


class SlotShow:
    def __init__(self, res_slot=[10, 15], avial_slot=[[0, 1], [2, 1], [7, 10], [0, 1], [2, 1], [7, 10]]):
        self.res_slot = res_slot
        self.avail_slot = np.asarray(avial_slot)
        self.percent_slot = (
            self.avail_slot / self.res_slot * 8).round().astype(int)

    def compute_chart(self):
        bar = " ▁▂▃▄▅▆▇█"
        bars = [char for char in bar]
        for i in range(len(self.res_slot)):
            bar_show = [bars[s] for s in self.percent_slot[:, i]]
            print("- Resources #", i, ":𝄃", ''.join(bar_show), "𝄃")


class Machine:
    def __init__(self, id=0, num_res=2, time_horizon=20, job_slot_size=10, job_backlog_size=10, res_slot=[20, 40], cost_vector=[4, 6], current_time=0) -> None:
        '''
        Initializes the machine
        '''
        self.id = id
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time
        self.job_slot = Job.JobSlot(job_slot_size)
        self.job_backlog = Job.JobBacklog(job_backlog_size)
        self.res_slot = res_slot
        self.reward = 0
        self.cost_vector = cost_vector
        self.avail_slot = np.ones((self.time_horizon, self.num_res))\
            * self.res_slot
        self.running_job = []
        self.reward = 0

    def get_price(self, job=Job.Job()):
        pass

    def allocate_job(self, job=Job.Job()):
        '''
            Allocate the Job to this Machine
        '''
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i:i+job.len, :]-job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True

                self.avail_slot[i:i+job.len] = new_avail_res
                job.start(self.current_time + i)
                job.finish(job.start_time + job.len)

                self.running_job.append(job)

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                break
        return allocated

    def time_proceed(self):
        '''
        process time
        '''

        self.avail_slot[:-1, :] = self.avail_slot[1:, :]
        self.avail_slot[-1, :] = self.res_slot

        for job in self.running_job:
            if job.finish_time <= self.current_time:
                self.reward += job.price
                self.running_job.remove(job)
        self.current_time += 1

    def show_running_job(self):
        print("Running Jobs:")
        print([job.id for job in self.running_job])

    def show_available_slot(self):
        print("Resource slots:")
        print(self.res_slot)
        print("Available slots:")
        slot_show = SlotShow(self.res_slot, self.avail_slot)
        slot_show.compute_chart()

    def show_res_vec(self):
        """
        Purpose: 
        """
        res_vec = [' '.join([str(c) for c in i]) for i in self.avail_slot.T]
        print("Resources Vector")
        for i in range(len(self.res_slot)):
            print("- Resources #", i, ":", res_vec[i])

    def show(self, verbose=False):
        """

        Purpose: show the state of this machine
        """

        table = prettytable.PrettyTable(
            ["id", "Current Time", "Number of Res", "Time Horizon", "Resource Slot", "Reward", "Cost Vector", "Number of Running Jobs"])
        table.add_row(
            [self.id, self.current_time,  self.num_res, self.time_horizon, self.res_slot, self.reward, self.cost_vector, str(len(self.running_job))])
        table.title = "Machine Info"
        print(table)
        self.show_available_slot()
        self.show_res_vec()
        self.show_running_job()
        print(self.reward)


class MachineSet:
    def __init__(self):
        self.number = 0
        self.machines = []

    def add_machine(self, num_res, time_horizon, job_slot_size, res_slot, cost_vector, current_time=0):
        self.machines.append(Machine(self.number,
                                     num_res, time_horizon, job_slot_size, res_slot, cost_vector, current_time))
        self.number += 1

    def generate_machines_random(self, num):
        """
        Purpose: 
        """
        for i in range(num):
            self.add_machine(num_res=2, time_horizon=20, job_slot_size=10, res_slot=[
                             20, 40], cost_vector=[4, 6], current_time=0)

    # end def
