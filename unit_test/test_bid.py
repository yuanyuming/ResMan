from networkx import identified_nodes


def test_bid():
    import Environment
    import Machine

    para = Environment.VehicleJobSchedulingParameters()
    a = 0
    for jobs in para.job_iterator:
        for job in jobs:
            print(job)
            bids = Machine.Bids(para.cluster, job)
            print(bids)
        if a == 100:
            break
        a += 1
    pass


def test_restrict():
    import Environment
    import Machine

    para = Environment.VehicleJobSchedulingParameters()
    for jobs in para.job_iterator:
        for job in jobs:
            print(job)
            for id in job.restrict_machines:
                print(id)
                print(para.cluster.machines[id])
        break
    pass


def test_get_machine():
    import Environment
    import Machine

    para = Environment.VehicleJobSchedulingParameters()
    for jobs in para.job_iterator:
        for job in jobs:
            print(job)
            for id in job.restrict_machines:
                print("Machine ID", id)
                print(para.cluster.get_machine(id).id)
        break
    pass
