import ray

from rllib_setup import get_env_continuous


def test_env():
    ray.init()
    env = get_env_continuous()
    env_name = "VJS"
    ray.tune.registry.register_env(env_name, lambda config: env)
    ray.rllib.utils.check_env(env)
