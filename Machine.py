import numpy as np
import prettytable
import Job


class SlotShow:
    def __init__(
        self,
        res_slot=[10, 15],
        avial_slot=[[0, 1], [2, 1], [7, 10], [0, 1], [2, 1], [7, 10]],
    ):
        self.res_slot = res_slot
        self.avail_slot = np.asarray(avial_slot)
        self.percent_slot = (
            self.avail_slot / self.res_slot * 8).round().astype(int)

    def compute_chart(self):
        bar = " ▁▂▃▄▅▆▇█"
        bars = [char for char in bar]
        for i in range(len(self.res_slot)):
            bar_show = [bars[s] for s in self.percent_slot[:, i]]
            print("- Resources #", i, ":𝄃", "".join(bar_show), "𝄃")


class BiderPolicy:
    def __init__(self) -> None:
        pass

    def bid(self, job=Job.Job()):
        pass


class Machine:
    def __init__(
        self,
        id=0,
        num_res=2,
        time_horizon=20,
        job_slot_size=10,
        job_backlog_size=10,
        res_slot=[20, 40],
        cost_vector=[4, 6],
        current_time=0,
        policy=0
    ) -> None:
        """
        Initializes the machine
        """
        self.id = id
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time
        self.job_slot_size = job_slot_size
        self.job_backlog_size = job_backlog_size
        self.job_slot = Job.JobSlot(self.job_slot_size)
        self.job_backlog = Job.JobBacklog(self.job_backlog_size)
        self.res_slot = res_slot
        self.reward = 0
        self.earning = 0
        self.finished_job = []
        self.cost_vector = cost_vector
        self.avail_slot = np.ones(
            (self.time_horizon, self.num_res)) * self.res_slot
        self.res_slot_time = self.avail_slot
        self.running_job = []
        # Bid
        self.request_job = Job.Job()
        self.policy = self.drl_bid
        self.action = 1
        self.bid = 0

    def add_backlog(self, job=Job.Job()):
        """
        Add the Job to the backlog
        """
        self.job_backlog.add_job(job)

    def reset(self):
        self.job_slot = Job.JobSlot(self.job_slot_size)
        self.job_backlog = Job.JobBacklog(self.job_backlog_size)
        self.earning = 0
        self.finished_job = []
        self.avail_slot = np.ones(
            (self.time_horizon, self.num_res)) * self.res_slot
        self.running_job = []
        self.request_job = None

    def get_bid(self):
        if self.request_job is not None:
            return self.policy(self.request_job)
        return 0

    def drl_bid(self, job=Job.Job()):
        return self.action*(np.dot(job.res_vec, self.cost_vector)) * job.len

    def set_action(self, action=1):
        self.action = action

    def fixed(self, job=Job.Job()):
        return np.dot(job.res_vec, self.cost_vector)

    def clear_job(self):
        self.request_job = None

    def observe(self):
        if self.request_job is not None:
            machine_obs = {'avail_slot': self.avail_slot,
                           'request_job': self.request_job.observe()}
            return machine_obs
        machine_obs = {'avail_slot': self.avail_slot, 'request_job': {'res_vec': [0, 0], 'len': 0,
                                                                      'priority': 0}}
        return machine_obs

    def fixed_norm(self, job=Job.Job(), var=0.2):
        return (
            (np.dot(job.res_vec, self.cost_vector) + var * np.random.normal())
            * job.len
            * job.priority
        )

    def request_auction(self, job=Job.Job()):
        self.request_job = job
    # async allocate_job, not used

    def can_allocate(self, job=Job.Job()):
        """
        Check if the Job can be allocated to this Machine
        """
        allocated = False
        new_avail_res = self.avail_slot[0: job.len, :] - job.res_vec
        if np.all(new_avail_res[:] >= 0):
            allocated = True
        return allocated
    # async allocate_job, not used

    def can_allocate_async(self, job=Job.Job()):
        """
        Check if the Job can be allocated to this Machine
        """
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i: i + job.len, :] - job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True
                break
        return allocated

    def allocate_job(self, job=Job.Job()):
        """
        Allocate the Job to this Machine
        """
        allocated = False
        new_avail_res = self.avail_slot[0: job.len, :] - job.res_vec
        if np.all(new_avail_res[:] >= 0):
            allocated = True
            self.avail_slot[0: job.len] = new_avail_res
            job.start(self.current_time)
            self.running_job.append(job)
            assert job.start_time != -1
            assert job.finish_time != -1
            assert job.finish_time > job.start_time
        return allocated
    # async allocate job, not used

    def allocate_job_async(self, job=Job.Job()):
        """
        Allocate the Job to this Machine
        """
        allocated = False

        for i in range(0, self.time_horizon - job.len):
            new_avail_res = self.avail_slot[i: i + job.len, :] - job.res_vec
            if np.all(new_avail_res[:] >= 0):
                allocated = True

                self.avail_slot[i: i + job.len] = new_avail_res
                job.start(self.current_time + i)

                self.running_job.append(job)

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                break
        return allocated

    def step(self):
        """
        process time
        """
        self.reward = 0

        self.avail_slot[:-1, :] = self.avail_slot[1:, :]
        self.avail_slot[-1, :] = self.res_slot

        for job in self.running_job:
            if job.finish_time <= self.current_time:
                self.reward += job.pay
                self.finished_job.append(job)
                self.running_job.remove(job)
        self.current_time += 1
        self.earning += self.reward
        self.request_job = None
        return self.reward

    def show_running_job(self):
        print("Running Jobs:")
        print([job.id for job in self.running_job])

    def show_available_slot(self):
        print("Machine #", self.id)
        print("Resource slots:")
        print(self.res_slot)
        print("Available slots:")
        slot_show = SlotShow(self.res_slot, self.avail_slot)
        slot_show.compute_chart()

    def show_res_vec(self):
        """
        Purpose:
        """
        res_vec = [" ".join([str(c) for c in i]) for i in self.avail_slot.T]
        print("Resources Vector")
        for i in range(len(self.res_slot)):
            print("- Resources #", i, ":", res_vec[i])

    def show(self, verbose=False):
        """

        Purpose: show the state of this machine
        """

        table = prettytable.PrettyTable(
            [
                "id",
                "Current Time",
                "Number of Res",
                "Time Horizon",
                "Resource Slot",
                "Reward",
                "Cost Vector",
                "Number of Running Jobs",
            ]
        )
        table.add_row(
            [
                self.id,
                self.current_time,
                self.num_res,
                self.time_horizon,
                self.res_slot,
                self.reward,
                self.cost_vector,
                str(len(self.running_job)),
            ]
        )
        table.title = "Machine Info"
        print(table)
        self.show_available_slot()
        self.show_res_vec()
        self.show_running_job()
        print(self.reward)

    def __str__(self) -> str:
        table = prettytable.PrettyTable(
            [
                "id",
                "Current Time",
                "Number of Res",
                "Time Horizon",
                "Resource Slot",
                "Reward",
                "Cost Vector",
                "Number of Running Jobs",
            ]
        )
        table.add_row(
            [
                self.id,
                self.current_time,
                self.num_res,
                self.time_horizon,
                self.res_slot,
                self.reward,
                self.cost_vector,
                str(len(self.running_job)),
            ]
        )
        table.title = "Machine Info"
        print(table)
        self.show_available_slot()
        self.show_res_vec()
        self.show_running_job()
        print(self.reward)
        return table.get_string() + "\n" + str(self.reward)


class Cluster:
    def __init__(
        self,
        machine_numbers=20,
        job_backlog_size=10,
        job_slot_size=10,
        num_res=2,
        time_horizon=20,
        current_time=0,
        machine_average_res_vec=[20, 40],
        machine_average_cost_vec=[2, 4],
        bias_r=5,
        bias_c=2,
    ):
        self.number = machine_numbers
        self.job_backlog_size = job_backlog_size
        self.job_slot_size = job_slot_size
        self.num_res = num_res
        self.time_horizon = time_horizon
        self.current_time = current_time
        self.now_id = 0
        self.machine_average_res_vec = machine_average_res_vec
        self.machine_average_cost_vec = machine_average_cost_vec
        self.bias_r = bias_r
        self.bias_c = bias_c
        self.machines = []
        self.generate_machines_random(self.number)

    def add_machine(self, res_slot, cost_vector):
        self.machines.append(
            Machine(
                self.now_id,
                self.num_res,
                self.time_horizon,
                self.job_slot_size,
                self.job_backlog_size,
                res_slot,
                cost_vector,
                self.current_time,
            )
        )
        self.now_id += 1

    def generate_machines_random(self, num):
        """
        Purpose:
        """
        for i in range(num):
            bias_r = np.random.randint(-self.bias_r, self.bias_r, self.num_res)
            bias_c = np.random.randint(-self.bias_c, self.bias_c, self.num_res)
            self.add_machine(
                res_slot=self.machine_average_res_vec + bias_r,
                cost_vector=self.machine_average_cost_vec + bias_c,
            )

    def allocate_job(self, machine_id=0, job=Job.Job()):
        self.machines[machine_id].allocate_job(job)

    def step(self):
        reward = [machine.step() for machine in self.machines]
        self.current_time += 1
        return reward

    def observe(self):
        return [machine.observe() for machine in self.machines]

    def clear_job(self):
        for machine in self.machines:
            machine.clear_job()

    def reset(self):
        self.current_time = 0
        for machine in self.machines:
            machine.reset()

    def get_machine(self, machine_id):
        return self.machines[machine_id]

    def show(self):
        table = prettytable.PrettyTable(
            ["id", "Resource Slot", "Reward", "Cost Vector"]
        )
        for machine in self.machines:
            table.add_row(
                [machine.id, machine.res_slot, machine.reward, machine.cost_vector]
            )
            machine.show_available_slot()
        print(table)


class Bids:
    def __init__(self, cluster=Cluster(), job=Job.Job()):
        self.machines = [cluster.get_machine(i) for i in job.restrict_machines]
        self.job = job
        self.can_allocate = []
        self.bids = []

    def get_bids(self):
        for machine in self.can_allocate:
            self.bids.append(machine.get_bid())

    def request_bids(self):
        for machine in self.machines:
            if machine.can_allocate(self.job):
                machine.request_auction(self.job)
                self.can_allocate.append(machine)

    def __str__(self):
        table = prettytable.PrettyTable(
            ["Machine " + str(machine.id) for machine in self.machines]
        )
        table.add_row(self.bids)
        return table.get_string()


class JobCollection:
    def __init__(self, collection=[Job.Job()]) -> None:
        self.collection = collection


# 将JobCollection的迭代器传入MachineRestrict的迭代器,返回一个迭代器


class MachineRestrict:
    def __init__(
        self,
        cluster=Cluster(),
        collection=Job.JobCollection(),
        max_machines=10,
        min_machines=3,
    ) -> None:
        self.cluster = cluster
        self.collections = collection
        self.iter_collection = iter(collection)
        self.collection = []
        self.max_machines = max_machines
        self.min_machines = min_machines

    def reset(self):
        self.collections.reset()
        self.iter_collection = iter(self.collections)
        self.collection = []

    def generate_restrict(self):
        # 生成限制机器,首先随机生成一个机器的下标,然后随机生成一个机器的数量,打乱机器的顺序,取前面的机器
        #
        collection = next(self.iter_collection)
        for jobs in collection:
            for job in jobs:
                min_machine_num = np.random.randint(
                    0, self.cluster.number - self.max_machines
                )
                t = np.random.randint(self.min_machines, self.max_machines)
                array = np.arange(
                    min_machine_num, min_machine_num + self.max_machines+1)
                np.random.shuffle(array)
                # TODO 限制机器的数量,验证是否有空槽位
                job.restrict_machines = array[:t]
        self.collection = collection
        return collection

    def show(self):
        table = prettytable.PrettyTable(
            ["Job Id", "Enter Time", "Restrict Machine"])
        for jobs in self.collection:
            for job in jobs:
                table.add_row([job.id, job.enter_time, job.restrict_machines])
        print(table)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_restrict()


class ListIterator:
    def __init__(self, iterator):
        self.iterator = iterator  # 输入的迭代器
        self.list = []  # 存储列表的变量
        self.index = 0  # 当前列表的索引

    def __iter__(self):
        return self  # 返回自身作为迭代器

    def __next__(self):
        if self.index >= len(self.list):  # 如果索引超出了列表的长度
            self.list = next(self.iterator)  # 调用输入的迭代器产生一个新的列表
            self.index = 0  # 重置索引
        element = self.list[self.index]  # 获取当前列表的元素
        self.index += 1  # 索引加一
        return element  # 返回元素


# 定义一个可迭代的类
class NestedList:
    # 初始化方法，接受一个列表的列表作为参数
    def __init__(self, nested_list):
        # 把参数赋值给实例属性
        self.nested_list = iter(nested_list)
        # 初始化当前的子列表和索引
        self.sublist = next(self.nested_list)
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
            self.quotes.append((self.cluster[machine_id].get_price(self.job)))
        return self.quotes

    def get_pay(self):
        arr = np.array(self.quotes)
        min_value = np.min(arr)
        min_index = np.argmin(arr)
        self.job.pay = min_value
        self.job.running_machine = self.job.restrict_machines[min_index]

    def show(self):
        table = prettytable.PrettyTable(["Machine", "Price"])
        for item in self.quotes:
            table.add_row([*item])

        print(table)
