def test_auction():
    import Environment
    para = Environment.VehicleJobSchedulingParameters()
    i,j = 0,0
    for jobs in para.job_iterator:
        for job in jobs:
            print(job)
            j+=1
            if para.auction_type.auction(job):
                i+=1
            print(job.pay)
            print(job.running_machine)
            # print(job)
            # print(para.cluster.machines[job.running_machine])
        para.cluster.step()
        if i==100:
            break
    print(i," ",j)
    para.cluster.show()
