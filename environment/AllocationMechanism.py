import numpy as np

from . import Machine


class AllocationMechanism:
    """Base class for allocation mechanisms"""

    def __init__(self):
        pass

    def allocate(self, bids):
        winners = np.argsort(bids.bids)[0]
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        second_prices = sorted_bids[1]
        return bids.machines[winners], prices, second_prices


class FirstPrice(AllocationMechanism):
    """(Generalised) First-Price Allocation"""

    def __init__(self):
        super().__init__()

    def allocate(self, bids):
        winners = int(np.argsort(bids.bids)[0])
        sorted_bids = np.sort(bids.bids)
        prices = sorted_bids[0]
        if len(sorted_bids) == 1:
            second_prices = prices
        else:
            second_prices = sorted_bids[1]
        return bids.can_allocate[winners], prices, second_prices

    def find_second_price(self, bids):
        winners = np.argmin(bids.bids)
        prices = bids.bids[winners]
        second_prices = np.partition(bids.bids, 1)[1]
        return bids.machines[winners], prices, second_prices


class SecondPrice(AllocationMechanism):
    """(Generalised) Second-Price Allocation"""

    def __init__(self):
        super().__init__()

    def allocate(self, bids):
        if len(bids.bids) == 1:
            winners = 0
            prices = bids.bids[0]
            # second_prices = prices
        else:
            winners = np.argsort(bids.bids)[0]
            prices = np.sort(bids.bids)[1]
        return bids.can_allocate[winners], prices, prices
