import AllocationMechanism
import Job
import Machine
import numpy as np


class ReverseAuction:
    def __init__(
        self,
        cluster=Machine.Cluster(),
        allocation_mechanism=AllocationMechanism.AllocationMechanism(),
    ) -> None:
        self.cluster = cluster
        self.allocation_mechanism = allocation_mechanism
        pass

    def auction(self, job=Job.Job()):
        bids = Machine.Bids(self.cluster, job)
        winner_machine, prices, second_prices = self.allocation_mechanism.allocate(bids)
        if prices > bids.job.budget:
            bids.job.pay = 0
            return False
        bids.job.pay = prices
        winner_machine.allocate_job(bids.job)
        return True


# 接收请求
# 找出可以执行的服务器
# 将请求发给服务器
# 服务器返回报价
# 确定胜者和支付
# 判断是否获胜
