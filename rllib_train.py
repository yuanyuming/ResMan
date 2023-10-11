import argparse

import ray

import rllib_a3c
import rllib_ddpg
import rllib_dqn
import rllib_ppo
import rllib_sac

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=int, default=30)
parser.add_argument("--machine", type=int, default=12)
parser.add_argument("--algo", type=str, default="ppo")
parser.add_argument("--rollout", type=int, default=10)
parser.add_argument("--allocation", type=str, default="FirstPrice")
args = parser.parse_args()

ray.init()

if args.algo == "ppo":
    rllib_ppo.train(args.jobs, args.machine, args.rollout, args.allocation)
elif args.algo == "ddpg":
    rllib_ddpg.train(args.jobs, args.machine)
elif args.algo == "sac":
    rllib_sac.train(args.jobs, args.machine)
elif args.algo == "a3c":
    rllib_a3c.train(args.jobs, args.machine)
elif args.algo == "dqn":
    rllib_dqn.train(args.jobs, args.machine)
else:
    print("algo not supported")
