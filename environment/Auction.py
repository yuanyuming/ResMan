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

    def auction(self, job=Job.Job()):
        self.current_job = job
        self.bids = Machine.Bids(self.cluster, self.current_job)

        self.bids.get_bids()
        bids = self.bids

        # 拍卖失败
        if len(bids.can_allocate) == 0:
            bids.job.pay = 0
            bids.job.running_machine = -1
            return False
        winner_machine, prices, _ = self.allocation_mechanism.allocate(bids)
        if prices > bids.job.budget:
            bids.job.pay = 0
            bids.job.running_machine = -1
            # print("Auction Failed: " + str(bids.job))
            return False
        # 拍卖成功
        bids.job.start(prices, winner_machine.id)
        winner_machine.allocate_job(bids.job)
        # print([mac.id for mac in bids.machines])
        # print(bids)
        # print(
        #    "Auction Success: "
        #    + "Job:"
        #    + str(bids.job.id)
        #    + " Pay:"
        #    + str(prices)
        #    + " Machine:"
        #    + str(winner_machine.id)
        # )
        return True


# 接收请求
# 找出可以执行的服务器
# 将请求发给服务器
# 服务器返回报价
# 确定胜者和支付
# 判断是否获胜
