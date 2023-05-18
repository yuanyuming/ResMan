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

        machine.step()
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
    
def test_env_action():
    import Environment
    env = Environment.VehicleJobSchedulingEnv()
    action = env.action_space('0')
    print("action space:")
    print(action.sample())

def test_env_obs():
    import Environment
    env = Environment.VehicleJobSchedulingEnv()
    obs = env.observation_space('0')
    print("observation space:")
    print(obs.sample())

def init_env():
    import Environment
    env = Environment.VehicleJobSchedulingEnv()
    return env
def get_jobs_iter():
    import Environment
    para = Environment.VehicleJobSchedulingParameters()
    return para.job_iterator

def jobs_per_round():
    iter = get_jobs_iter()
    for i in range(100):
        jobs = next(iter)
        print(len(jobs))
        for job in jobs:
            yield job,False
        print("round: " + str(i))
        yield None,True
        
        
def test_env_step():
    env = init_env()
    i = 0
    for job ,done in jobs_per_round():
        i+=1
        if done:
            env.parameters.cluster.step()
            continue
        actions = {agent: env.action_space(
            agent).sample() for agent in env.agents}
        env.step(actions)
        env.parameters.auction_type.auction(job)
        if i == 1000:
            break
    env.parameters.cluster.show()
    
def test_env():
    env = init_env()
    from pettingzoo.test import parallel_api_test
    parallel_api_test(env, num_cycles=1000)

        
def my_generator(n):
    index = 0
    while index < n:
        yield index
        index += 1
    yield None

def test_generator():
    ge = my_generator(2)
    print(next(ge))
    print(next(ge))
    print(next(ge))
