# TODO

mappo modify

- [ ] priority

## Policy To Estimate

- [ ] TruthfulPolicy
- [ ] EmpiricalPolicy
- [ ] ValueLearningPolicy
- [ ] PolicyLearningPolicy
- [ ] DoublyRobustPolicy
- [ ] DQNPolicy
- [ ] A2C
- [ ] A3C
- [ ] MAPPOPolicy
- [ ] MADDPGPolicy
- [ ] SACPolicy

## Estimate Setup

- [ ] Job Saver
- [ ] Job Loader
- [ ] Machine Loader
- [ ] Policy Loader

# Eassy

cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 15 --machine 6
cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 30  --machine 6
cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 30 --machine 12
cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 60 --machine 12
cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 60 --machine 24
cd ResMan
source man/bin/activate
python rllib_train.py --algo ddpg --job 120 --machine 24
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 15 --machine 6
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 30  --machine 6
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 30 --machine 12
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 60 --machine 12
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 60 --machine 24
cd ResMan
source man/bin/activate
python rllib_train.py --algo a3c --job 120 --machine 24
