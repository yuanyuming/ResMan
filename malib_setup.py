import importlib
from typing import Any, Dict, List, Sequence, Tuple, Union

import gym
import pettingzoo
from pettingzoo.utils.wrappers import BaseWrapper

from environment.Environment import VehicleJobSchedulingEnvACE
from malib.rollout.envs.env import Environment
from malib.utils.typing import AgentID


class PettingZooEnv(Environment):
    def __init__(self, env=BaseWrapper(VehicleJobSchedulingEnvACE()), **configs):
        super(PettingZooEnv, self).__init__(**configs)

        # env_id = configs["env_id"]
        self.env = env
        parallel_simulate = False
        # scenario_configs = configs["scenario_configs"].copy()
        # parallel_simulate = scenario_configs.pop("parallel_simulate", False)

        # domain_id, scenario_id = env_id.split(".")

        # domain = importlib.import_module(f"pettingzoo.{domain_id}")
        # scenario = getattr(domain, scenario_id)

        # if parallel_simulate:
        #     self.env: pettingzoo.ParallelEnv = scenario.parallel_env(**scenario_configs)
        # else:
        #     self.env: pettingzoo.AECEnv = scenario.env(**scenario_configs)

        self._parallel_simulate = parallel_simulate
        self._action_spaces = {
            agent: self.env.action_space(agent) for agent in self.env.possible_agents
        }
        self._observation_spaces = {
            agent: self.env.observation_space(agent)
            for agent in self.env.possible_agents
        }

    @property
    def possible_agents(self) -> List[AgentID]:
        return self.env.possible_agents.copy()

    @property
    def parallel_simulate(self) -> bool:
        return self._parallel_simulate

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        if not self.parallel_simulate:
            self.env.step(actions[self.env.agent_selection])
            rewards = self.env.rewards.copy()
            observations = {
                agent: self.env.observe(agent) for agent in self.possible_agents
            }
            dones = self.env.terminations.copy()
            infos = self.env.infos.copy()
        else:
            observations, rewards, dones, truncations, infos = map(
                dict, self.env.step(actions)
            )

        return None, observations, rewards, dones, infos

    def reset(self, max_step: int = None) -> Union[None, Sequence[Dict[AgentID, Any]]]:
        super(PettingZooEnv, self).reset(max_step)
        observations = self.env.reset()
        if observations is None:
            # means AECEnv
            observations = {
                agent: self.env.observe(agent) for agent in self.possible_agents
            }

        return None, observations

    def seed(self, seed: int = None):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render()

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    env = PettingZooEnv()
    action_spaces = env.action_spaces
    _, observations = env.reset()
    done = False
    cnt = 0
    while not done:
        actions = {k: action_spaces[k].sample() for k in observations.keys()}
        _, observations, rewards, dones, infos = env.step(actions)
        done = dones["__all__"]
        cnt += 1
    print("test passed after {} steps".format(cnt))
