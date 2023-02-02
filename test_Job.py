import pytest
import job_distribution
import numpy
import Parameters

def test_JobDist():
    """
    Purpose: 
    """
    nw_len_seq,nw_size_seq = job_distribution.generate_sequence_work()
    print(nw_len_seq,nw_size_seq)
# end def
