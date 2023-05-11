import numpy as np
import Machine


class AllocationMechanism:
    """Base class for allocation mechanisms"""

    def __init__(self):
        pass

    def allocate(self, bids=Machine.Bids()):
        winners = np.argsort(bids.bids)[0]
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        second_prices = sorted_bids[1]
        return bids.machines[winners], prices, second_prices


class FirstPrice(AllocationMechanism):
    """(Generalised) First-Price Allocation"""

    def __init__(self):
        super(FirstPrice, self).__init__()

    def allocate(self, bids=Machine.Bids()):
        winners = np.argsort(bids.bids)[0]
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        second_prices = sorted_bids[1]
        return bids.machines[winners], prices, second_prices


class SecondPrice(AllocationMechanism):
    """(Generalised) Second-Price Allocation"""

    def __init__(self):
        super(SecondPrice, self).__init__()

    def allocate(self, bids=Machine.Bids()):
        winners = np.argsort(bids.bids)[0]
        prices = np.sort(bids.bids)[1]
        return bids.machines[winners], prices, prices
