
import Environment
import Job
import sys
sys.path.append(".")


def test_jobSlot():
    """
    Purpose: make sure that JobSlot works properly
    """
    env = Environment.Allocation_Environment()
    job = env.get_new_job_from_seq(0, 0)
    job.show()
    job_slot = Job.JobSlot(10)
    job_slot.slot = [job] * 10
    job_slot.show()


def test_jobbacklog():
    """
    Purpose: test that JobBacklog works properly
    """
    env = Environment.Allocation_Environment()
    job = env.get_new_job_from_seq(0, 0)
    job.show()
    jobbacklog = Job.JobBacklog(10)
    jobbacklog.backlog = [job] * 10
    jobbacklog.curr_size = 10
    jobbacklog.show()
