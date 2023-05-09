def test_step():
    import Environment
    import numpy as np

    para = Environment.VehicleJobSchedulingParameters()
    cluster = para.cluster
    for i in range(100):
        jobs = next(para.job_iterator)
        print(len(jobs))
        for job in jobs:
            cluster.allocate_job(machine_id=np.random.randint(0, 9), job=job)
        cluster.step()
    print(cluster.machines[0])


def test_env_single_para():
    import Environment_single
    import numpy as np

    para = Environment_single.Parameters()
    machine = para.machine
    all_jobs = 0
    allocate_jobs = 0
    for i in range(10):
        jobs = next(para.job_iterator)
        for job in jobs:
            all_jobs += 1
            if machine.allocate_job(job=job):
                print("Success" + str(job))
                allocate_jobs += 1

        machine.time_proceed()
    print(machine)
    print(
        "All Jobs: "
        + str(all_jobs)
        + ", Allocate Jobs: "
        + str(allocate_jobs)
        + ", Allocate Rate: "
        + str(allocate_jobs / all_jobs)
    )


def get_job():
    import Environment

    para = Environment.VehicleJobSchedulingParameters()
    return next(para.job_iterator)[0]


def test_job_show():
    get_job().show()
    print(get_job())
