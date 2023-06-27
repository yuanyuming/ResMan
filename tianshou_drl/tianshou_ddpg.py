import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from torch.utils.tensorboard import SummaryWriter


class Args:
    def __init__(
        self,
        task="Pendulum-v1",
        reward_threshold=None,
        seed=0,
        buffer_size=20000,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        exploration_noise=0.1,
        epoch=5,
        step_per_epoch=20000,
        step_per_collect=8,
        update_per_step=0.125,
        batch_size=128,
        hidden_sizes=[128, 128],
        training_num=8,
        test_num=100,
        logdir="log",
        render=0.0,
        rew_norm=False,
        n_step=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.task = task
        self.reward_threshold = reward_threshold
        self.seed = seed
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.epoch = epoch
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect
        self.update_per_step = update_per_step
        self.batch_size = batch_size
        self.hidden_sizes = hidden_sizes
        self.training_num = training_num
        self.test_num = test_num
        self.logdir = logdir
        self.render = render
        self.rew_norm = rew_norm
        self.n_step = n_step
        self.device = device


# 假设这是一个用于训练和评估DDPG算法的类
class DDPGTrainer:
    def __init__(self, get_env, args=Args()):
        self.get_env = get_env
        self.args = args
        self.policy, self.agents = self._get_agents()
        self.train_collector, self.test_collector = self.construct_collector()
        # log
        log_path = os.path.join(args.logdir, args.task, "ddpg")
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        # callback function
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        self.save_best_fn = save_best_fn
        self.stop_fn = stop_fn
        self.logger = logger
        self.result = None

    def _get_agents(self):
        env = self.get_env()
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]
        agents = []
        for i in range(env.num_agents):
            # model
            net = Net(
                state_shape,
                hidden_sizes=self.args.hidden_sizes,
                device=self.args.device,
            )
            actor = Actor(
                net, action_shape, max_action=max_action, device=self.args.device
            ).to(self.args.device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.actor_lr)
            net = Net(
                state_shape,
                action_shape,
                hidden_sizes=self.args.hidden_sizes,
                concat=True,
                device=self.args.device,
            )
            critic = Critic(net, device=self.args.device).to(self.args.device)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=self.args.critic_lr)
            policy = DDPGPolicy(
                actor,
                actor_optim,
                critic,
                critic_optim,
                tau=self.args.tau,
                gamma=self.args.gamma,
                exploration_noise=GaussianNoise(sigma=self.args.exploration_noise),
                reward_normalization=self.args.rew_norm,
                estimation_step=self.args.n_step,
                action_space=env.action_space,
            )
            agents.append(policy)
        policy = MultiAgentPolicyManager(agents, env)
        return policy, agents

    def construct_collector(self):
        env = self.get_env()
        # you can also use tianshou.env.SubprocVectorEnv
        train_envs = DummyVectorEnv(
            [self.get_env for _ in range(self.args.training_num)]
        )
        test_envs = DummyVectorEnv([self.get_env for _ in range(self.args.test_num)])
        # seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        train_envs.seed(self.args.seed)
        test_envs.seed(self.args.seed)

        # collector
        train_collector = Collector(
            self.policy,
            train_envs,
            VectorReplayBuffer(self.args.buffer_size, len(train_envs)),
            exploration_noise=True,
        )
        test_collector = Collector(self.policy, test_envs)
        return train_collector, test_collector

    def train(self):
        # trainer
        self.result = offpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            self.args.epoch,
            self.args.step_per_epoch,
            self.args.step_per_collect,
            self.args.test_num,
            self.args.batch_size,
            update_per_step=self.args.update_per_step,
            stop_fn=self.stop_fn,
            save_best_fn=self.save_best_fn,
            logger=self.logger,
        )
        assert self.stop_fn(self.result["best_reward"])

    def eval(self):
        pprint.pprint(self.result)
        # Let's watch its performance!
        env = self.get_env()
        self.policy.eval()
        collector = Collector(self.policy, env)
        result = collector.collect(n_episode=1, render=self.args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


# 使用这个类的示例代码
if __name__ == "__main__":
    # 创建一个环境函数，例如gym.make("Pendulum-v0")
    get_env = lambda: gym.make("Pendulum-v0")
    # 创建一个参数对象，可以设置一些超参数，例如hidden_sizes=[128, 128], actor_lr=1e-3等
    args = Args(hidden_sizes=[128, 128], actor_lr=1e-3)
    # 创建一个DDPGTrainer对象，传入环境函数和参数对象
    trainer = DDPGTrainer(get_env, args)
    # 调用train方法进行训练
    trainer.train()
    # 调用eval方法进行评估
    trainer.eval()


if __name__ == "__main__":
    pass
