import numpy as np
import gym
import cartpole_swingup_envs
from pilco.models import PILCO
from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
import os
import random

SEED = 0

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from utils import rollout

SUBS = 1  # subsampling rate
bf = 50  # the number of basis functions used
maxiter= 50  # max iteration for model and policy optimization
max_action=1.0  # the maximum possible value that action can take

# Hyper-parameters of the reward function (this is tricky to tune!)
target = np.array([0.0, 0.0, np.pi, 0.0])
weights = np.diag([0.01, 0.1, 10.0, 0.1])

# Initial parameters of the GP model of the environment
m_init = np.reshape([0.0, 0.0, -1.0, 0.0], (1,4))
S_init = np.diag([0.05, 0.05, 0.01, 0.01])

T = 50  # the number of timesteps in each random rollout
T_sim = T  # the number of timesteps in each rollout that uses the controller
J = 30  # the number of random rollouts at the beginning before first optimization starts
N = 3  # the number of rollouts after first optimization (at this stage optimization is performed after each rollout)
restarts = 1 # the number of times that optimizations with different initializations are performed at each optimization step
print(J, N)

env_to_use = 'CartPole-v1'
print('ENV TO USE', env_to_use)
env = gym.make(env_to_use)
env.seed(SEED)
env.action_space.seed(SEED)

########################################################

# Initial random rollouts to generate a dataset
X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=False, render=False)

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)


checkpoint = tf.train.Checkpoint(pilco=pilco)
manager = tf.train.CheckpointManager(checkpoint, './MTests', max_to_keep=1)
#tf.saved_model.save(pilco.mgpr, './models')

# from IPython import embed; embed()


checkpoint.restore(manager.latest_checkpoint)

# input("Press enter to test")
# _ = rollout(env, pilco, timesteps=1000, verbose=False, SUBS=SUBS, render=True)


totalSteps = 0
extracted = []
for i in range(10000):
    # from IPython import embed; embed()
    _, _, _, _, states_and_dec  = rollout(env, pilco, timesteps=1000, verbose=False, SUBS=SUBS, render=False, extract=True)
    extracted.append(states_and_dec)

print('ASDFASD')
np.save('10000_reg', extracted)

# from IPython import embed; embed()
