import tianshou_setup
from tianshou_drl import tianshou_ddpg

ddpg = tianshou_ddpg.DDPGTrainer(tianshou_setup.get_env_continous)
ddpg.train()
