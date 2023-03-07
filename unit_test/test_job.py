
import Environment
import Job
import sys
sys.path.append(".")


def test_job():
    job = Job.Job()
    job.random_job()
    job.show()


def test_job_distribution():
    jD = Job.JobDistribution()
    l, ds = jD.normal_dist()
    print("len:", l, "\nSize:", ds)


def test_jobslot_add():
    job_slot = Job.JobSlot(10)
    job = Job.Job(job_id=11)
    job.random_job()
    job.show()
    job_slot.add_new_job(job=job)
    # job_slot.slot[2] = job
    job_slot.show()


def test_jobSlot():
    """
    Purpose: make sure that JobSlot works properly
    """
    job_slot = Job.JobSlot(10)
    for i in range(10):
        job = Job.Job(job_id=i)
        job.random_job()
        job_slot.add_new_job(job)
    print("Job slot:#10")
    job_slot.show()
    print("Job slot:Select job #4!")
    job_slot.select_job(4)
    job_slot.show()
    print('Job slot:New #11 Comes')
    job = Job.Job(job_id=11)
    job.random_job()
    job_slot.add_new_job(job)
    job_slot.show()


def test_jobbacklog():
    """
    Purpose: test that JobBacklog works properly
    """
    job_backlog = Job.JobBacklog(10)
    # Fill the log
    for i in range(10):
        job = Job.Job(job_id=i)
        job.random_job()
        job_backlog.add_backlog(job)
    job_backlog.show()
    # Refill the log
    for i in range(10, 20):
        job = Job.Job(job_id=i)
        job_backlog.add_backlog(job)
    job_backlog.show()


def test_jobrecord():
    """
    Purpose: 
    """
    job_record = Job.JobRecord()
    for i in range(100):
        job = Job.Job(job_id=i)
        job.random_job()
        job_record.new_record(job)

    job_record.show()

# end def
