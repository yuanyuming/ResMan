'''
Class: Machine
def:
    __init__
    get_color
    allocate_job
    time_proceed
    show
'''


import imp


def test_Machine_init():
    """
    Purpose: 
    """
    import Machine
    machine = Machine.Machine(1, 2, 20, 10, [25, 40], [4, 6])
    machine.show()
# end def


def test_caaa():
    """
    Purpose: 
    """
    import Machine
    ss = Machine.SlotShow()
    ss.compute_chart()
# end def


def test_machine_allocate():
    """
    Purpose: 
    """
    import Job
    import Machine
    import numpy as np
    jc = Job.JobCollection(average=15, duration=1000)
    collections = jc.get_job_collections()

    mac = Machine.Machine(1, 2, 20, 10, [20, 40], [4, 6])
    for collection in collections:
        for job in collection:
            job.price = np.random.randint(1, 10)
            mac.allocate_job(job)
        mac.time_proceed()
    mac.show()

# end def
