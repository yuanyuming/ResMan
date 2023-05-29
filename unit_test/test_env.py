from typing import Optional

import Environment


def test_step():
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
    import numpy as np

    import Environment_single

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
    action = env.action_space("0")
    print("action space:")
    print(action.sample())


def test_env_obs():
    from .. import Environment

    env = Environment.VehicleJobSchedulingEnv()
    obs = env.observation_space("0")
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
            yield job, False
        print("round: " + str(i))
        yield None, True


def test_env_step():
    env = init_env()
    i = 0
    for job, done in jobs_per_round():
        i += 1
        if done:
            env.parameters.cluster.step()
            continue
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.parameters.auction_type.auction(job)
        if i == 1000:
            break
    env.parameters.cluster.show()


def test_env():
    env = init_env()
    from pettingzoo.test import parallel_api_test

    parallel_api_test(env, num_cycles=1000)


def test_env_benchmark():
    import random
    import time

    import numpy as np

    env = init_env()
    print("Starting performance benchmark")
    cycles = 0
    turn = 0
    env.reset()
    start = time.time()
    end = 0
    rewards = np.zeros(len(env.agents))

    while True:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, reward, termination, truncation, info = env.step(actions)
        turn += 1
        rewards += np.array(list(reward.values()))
        if time.time() - start > 5:
            end = time.time()
            break

    length = end - start
    cycles = env.parameters.cluster.current_time
    turns_per_time = turn / length
    cycles_per_time = cycles / length
    print(str(turns_per_time) + " turns per second")
    print(str(cycles_per_time) + " cycles per second")
    print("Finished performance benchmark")
    print("Average reward per agent: " + str(rewards / cycles))
    print("Job Finish Rate: " + str(env.finished_job / env.total_job))


def test_job_per_step():
    env = init_env()
    job_generator = env.get_job_next_step()
    print(next(job_generator)[0])


def test_env_agent_selector():
    import Environment

    env = Environment.VehicleJobSchedulingEnvACE()
    env.reset()
    selector = env._agent_selector()
    for i in range(100):
        print(selector.next(), end=" ")
        if selector.is_last():
            env.request_job, _ = next(env.get_job)


def test_ace_env():
    from pettingzoo.test import api_test

    import Environment

    env = Environment.VehicleJobSchedulingEnvACE(render_mode="human")
    # env = Environment.VehicleJobSchedulingEnvACE()
    env.reset()
    api_test(env, num_cycles=1000)


def test_aec_preformce():
    env = Environment.VehicleJobSchedulingEnvACE()
    from pettingzoo.test import performance_benchmark

    performance_benchmark(env)


def test_ace_env_step():
    import Environment

    env = Environment.VehicleJobSchedulingEnvACE()
    env.reset()
    for agent in env.agent_iter(100):
        env.step(env.action_space(agent).sample())


def test_ace_env_observe():
    import Environment

    env = Environment.VehicleJobSchedulingEnvACE()
    env.reset()
    ob, *_ = env.last()
    print(ob)


import pettingzoo
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class reward_class(pettingzoo.AECEnv):
    def __init__(self):
        super().__init__()
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {
            "agent_0": spaces.Discrete(1),
            "agent_1": spaces.Discrete(1),
        }
        self.action_spaces = {
            "agent_0": spaces.Discrete(1),
            "agent_1": spaces.Discrete(1),
        }
        self.rewards = {"agent_0": 0, "agent_1": 0}
        self.dones = {"agent_0": False, "agent_1": False}
        self._agent_selector = agent_selector(self.agents)
        self.observation = {"agent_0": 0, "agent_1": 0}
        self.truncations = {"agent_0": False, "agent_1": False}
        self.terminations = {"agent_0": False, "agent_1": False}
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = {"agent_0": 0, "agent_1": 0}
        self.infos = {"agent_0": {}, "agent_1": {}}

    def reset(self, seed=None, return_info=False, options=None) -> None:
        self.rewards = {"agent_0": 0, "agent_1": 0}
        self.dones = {"agent_0": False, "agent_1": False}
        self.truncations = {"agent_0": False, "agent_1": False}
        self.terminations = {"agent_0": False, "agent_1": False}

    def step(self, action) -> None:
        agent = self.agent_selection

        self._clear_rewards()
        self.rewards[agent] = 1
        self.observation[agent] = int(action)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        print(self._cumulative_rewards)

    def observe(self, agent: str):
        return self.observation[agent]


def test_reward():
    env = reward_class()
    from pettingzoo.test import api_test

    api_test(env, num_cycles=10)
