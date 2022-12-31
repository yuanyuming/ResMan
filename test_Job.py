import pytest
import JobDistribution
import numpy


def test_JobDist():
    """
    Purpose: 
    """
    nw_len_seq,nw_size_seq = JobDistribution.generate_sequence_work()
    print(nw_len_seq,nw_size_seq)
# end def



import time
