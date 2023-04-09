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
        self.policy = self.fixed_norm

    def get_price(self, job=Job.Job()):
        return self.id, self.policy(job, self.cost_vector)

    def fixed(self, job=Job.Job(), cost_vector=[4, 6]):
        return np.dot(job.res_vec, cost_vector)

    def fixed_norm(self, job=Job.Job(), cost_vector=[4, 6], var=0.2):
        return np.dot(job.res_vec, cost_vector) + var * np.random.normal()

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
                self.reward += job.pay
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
    def __init__(self, machine=[Machine()]):
        self.machines = machine

    def add_machine(self, machine=Machine()):
        """
        Purpose: 
        """
        self.machines.append(machine)
    # end def


class Cluster:
    def __init__(self, machine_numbers=10, job_backlog_size=10, job_slot_size=10, num_res=2, time_horizon=20, current_time=0):
        self.number = machine_numbers
        self.machines = []
        self.job_backlog_size = job_backlog_size
        self.job_slot_size = job_slot_size
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time

    def add_machine(self, res_slot, cost_vector):
        self.machines.append(Machine(self.number, self.num_res, self.time_horizon,
                                     self.job_slot_size, self.job_backlog_size, res_slot, cost_vector, self.current_time))
        self.number += 1

    def generate_machines_random(self, num):
        """
        Purpose: 
        """
        for i in range(num):
            bais_r = np.random.randint(-5, 5)
            bais_c = np.random.randint(-2, 2)
            self.add_machine(
                res_slot=[20+bais_r, 40+bais_r], cost_vector=[4+bais_c, 6+bais_c])

    def allocate_job(self, machine_id=0, job=Job.Job()):
        self.machines[machine_id].allocate_job(job)

    def step(self):
        pass

    def show(self):
        table = prettytable.PrettyTable(
            ['id', "Resource Slot", "Reward", "Cost Vector"])
        for machine in self.machines:
            table.add_row([machine.id, machine.res_slot,
                          machine.reward, machine.cost_vector])
        print(table)


class JobCollection:
    def __init__(self, collection=[Job.Job()]) -> None:
        self.collection = collection

# 将JobCollection的迭代器传入MachineRestrict的迭代器,返回一个迭代器


class MachineRestrict:
    def __init__(self, cluster=Cluster(), collection=iter(Job.JobCollection()), max_machines=10, min_machines=3) -> None:
        self.cluster = cluster
        self.iter_collection = collection
        self.collection = []
        self.max_machines = max_machines
        self.min_machines = min_machines
        pass

    def generate_restrict(self):
        collection = next(self.iter_collection)
        for jobs in collection:
            for job in jobs:
                min = np.random.randint(
                    0, self.cluster.    number-self.max_machines)
                t = np.random.randint(self.min_machines, self.  max_machines)
                array = np.arange(min, min+self.max_machines)
                np.random.shuffle(array)
                job.restrict_machines = array[:t]
        self.collection = collection
        return collection

    def show(self):
        table = prettytable.PrettyTable(
            ['Job Id', 'Enter Time', 'Restrict Machine'])
        for jobs in self.collection:
            for job in jobs:
                table.add_row([job.id, job.enter_time, job.restrict_machines])
        print(table)

    def __iter__(self):

        return self

    def __next__(self):
        return self.generate_restrict()


# 定义一个可迭代的类
class NestedList:
    # 初始化方法，接受一个列表的列表作为参数
    def __init__(self, nested_list):
        # 把参数赋值给实例属性
        self.nested_list = iter(nested_list)
        # 初始化当前的子列表和索引
        self.sublist = []
        self.index = 0

    # 定义一个__iter__方法，返回一个迭代器对象
    def __iter__(self):
        # 返回自身作为迭代器对象
        return self

    # 定义一个__next__方法，返回下一个元素
    def __next__(self):

        # 如果当前的索引小于子列表的长度，就返回子列表中的元素，并增加索引
        if self.index < len(self.sublist):
            value = self.sublist[self.index]
            self.index += 1
            return value
        # 如果当前的索引等于子列表的长度，就抛出一个StopIteration异常，并清空子列表，以便下次迭代时获取下一个子列表
        else:
            self.index = 0
            self.sublist = next(self.nested_list)
            raise StopIteration


class Quote:
    def __init__(self, job=Job.Job(), cluster=Cluster()) -> None:
        self.job = job
        self.cluster = cluster.machines
        self.quotes = []
        self.get_price_set()
        pass

    def get_price_set(self):
        for machine_id in self.job.restrict_machines:
            self.quotes.append(
                (self.cluster[machine_id].get_price(self.job)))
        return self.quotes

    def get_pay(self):
        arr = np.array(self.quotes)
        min_value = np.min(arr)
        min_index = np.argmin(arr)
        self.job.pay = min_value
        self.job.running_machine = self.job.restrict_machines[min_index]

    def show(self):
        table = prettytable.PrettyTable(['Machine', 'Price'])
        for item in self.quotes:
            table.add_row([*item])

        print(table)
