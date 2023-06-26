"""
Class: Machine
def:
    __init__
    get_color
    allocate_job
    time_proceed
    show
"""
from environment import Job, Machine


def test_Machine_init():
    """
    Purpose:
    """
    import Machine

    machine = Machine.Machine(1, 2, 20, 10, 10, [25, 40], [4, 6])
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

    mac = Machine.Machine(1, 2, 20, 10, 10, [20, 40], [4, 6])
    for collection in collections:
        for job in collection:
            job.price = np.random.randint(1, 10)
            mac.allocate_job(job)
        mac.step()
    mac.show()


# end def


def test_machine_cluster():
    import Machine

    cluster = Machine.Cluster()
    cluster.generate_machines_random(10)
    cluster.show()


def test_strict_machine():
    cluster = Machine.Cluster()
    cluster.generate_machines_random(100)
    collection = Job.JobCollection(average=20)
    restrict = Machine.MachineRestrict(cluster=cluster, collection=iter(collection))
    next(restrict)
    restrict.show()
    next(restrict)
    restrict.show()


def test_strict_machine_iter():
    cluster = Machine.Cluster()
    cluster.generate_machines_random(100)
    collection = Job.JobCollection(average=20)
    mr = Machine.MachineRestrict(cluster, iter(collection))
    i = 0
    for collection in iter(mr):
        for jobs in collection:
            for job in jobs:
                job.show()
                print(i)
                break
            break
        i += 1
        if i == 1000:
            break


def test_job_iterator():
    cluster = Machine.Cluster()
    cluster.generate_machines_random(100)
    collection = Job.JobCollection(average=1)
    mr = Machine.MachineRestrict(cluster, iter(collection))
    it = Machine.NestedList(iter(mr))

    jt = Machine.NestedList(iter(it))
    for time in range(50):
        for job in jt:
            job.show()
        print("done!!!!!!!!!!!!!!!!!!!!!!", time)


def test_Zero():
    it = Machine.NestedList([[], [1, 2]])
    for job in it:
        print(job)
        print("!!!!!!!")
    print("1 done!!!!!!!")
    for job in it:
        print(job)


def test_policy_fixed():
    machine = Machine.Machine()
    price = machine.get_price()


def test_Quote():
    cluster = Machine.Cluster()
    cluster.generate_machines_random(30)
    jobs = Machine.MachineRestrict(cluster=cluster)
    quote = Machine.Quote(job=jobs.collection[0], cluster=cluster)
    quote.show()


def test_restrict():
    import Environment


def test_machine_observe():
    import Environment
    import Machine

    machine = Machine.Machine()
    machine.reset()
    print(machine.observe())
    env = Environment.VehicleJobSchedulingEnvACE()
    env.reset()
    env.step(0)
    env.observe()
