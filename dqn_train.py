import tianshou_setup
from tianshou_drl import tianshou_dqn

tianshou_dqn.train_dqn(tianshou_setup.get_env)
