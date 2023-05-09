import AllocationMechanism
import Machine
import numpy as np


class ReverseAuction:
    def __init__(self, allocation_mechanism=AllocationMechanism.FirstPrice()) -> None:
        self.allocation_mechanism = allocation_mechanism
        pass

    def auction(self, bids=Machine.Bids()):
        winner_machine, prices, second_prices = self.allocation_mechanism.allocate(bids)
        if prices > bids.job.budget:
            bids.job.pay = 0
            return False
        bids.job.pay = prices
        winner_machine.allocate_job(bids.job)
        return True
