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
    machine = Machine.Machine(1, 2, 20, [25, 40], [4, 6])
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
    jc = Job.JobCollection(average=15, duration=10)
    collections = jc.get_job_collections()

    mac = Machine.Machine(1, 2, 20, [25, 40], [4, 6])
    for collection in collections:
        for job in collection:
            mac.allocate_job(job)
    mac.show()

# end def
