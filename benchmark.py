import Environment

env = Environment.VehicleJobSchedulingEnvACE()
from pettingzoo.test import performance_benchmark

performance_benchmark(env)
