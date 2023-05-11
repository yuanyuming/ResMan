def test_auction():
    import Environment
    para = Environment.VehicleJobSchedulingParameters()
    for jobs in para.job_iterator:
        for job in jobs:
            print(job)
            para.auction.auction(job)
            print(job)
            print(para.cluster.machines[job.running_machine])
        break
