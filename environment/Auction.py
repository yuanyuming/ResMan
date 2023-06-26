import numpy as np

from . import AllocationMechanism, Job, Machine


class ReverseAuction:
    def __init__(
        self,
        cluster=Machine.Cluster(),
        allocation_mechanism=AllocationMechanism.AllocationMechanism(),
    ) -> None:
        self.cluster = cluster
        self.allocation_mechanism = allocation_mechanism
        self.current_job = Job.Job()
        self.bids = Machine.Bids(self.cluster, self.current_job)
        pass

    def auction(self, job=Job.Job()):
        self.request_auction(job)
        self.bids.get_bids()
        bids = self.bids
        if len(bids.can_allocate) == 0:
            bids.job.pay = 0
            bids.job.running_machine = -1
            return False
        winner_machine, prices, second_prices = self.allocation_mechanism.allocate(bids)

        if prices > bids.job.budget:
            bids.job.pay = 0
            bids.job.running_machine = -1
            return False
        bids.job.pay = prices
        bids.job.running_machine = winner_machine.id
        winner_machine.allocate_job(bids.job)
        return True

    def request_auction(self, job=Job.Job()):
        self.current_job = job
        self.bids = Machine.Bids(self.cluster, self.current_job)
        self.bids.request_bids()
        pass


# 接收请求
# 找出可以执行的服务器
# 将请求发给服务器
# 服务器返回报价
# 确定胜者和支付
# 判断是否获胜
