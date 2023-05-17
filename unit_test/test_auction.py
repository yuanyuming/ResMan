def test_auction():
    import Environment
    import tqdm
    para = Environment.VehicleJobSchedulingParameters()
    i,j = 0,0
    
    for _ in tqdm.tqdm(range(1000)):
        jobs = next(para.job_iterator)
        for job in jobs:
            # print(job)
            j+=1
            if para.auction_type.auction(job):
                i+=1
            # print(job.pay)
            # print(job.running_machine)
            # print(job)
            # print(para.cluster.machines[job.running_machine])
        para.cluster.step()
        if j==1000:
            break
    print(i," ",j)
    para.cluster.show()

def test_auction2():
    import Environment
    para = Environment.VehicleJobSchedulingParameters()
    i,j = 0,0
    for k in range(1000):
        jobs = next(para.job_iterator)
        for job in jobs:
            j+=1
            for machine in para.cluster.machines:
                if machine.allocate_job(job):
                    i+=1
                    break
        para.cluster.step()
    print(i," ",j)
    para.cluster.show()
