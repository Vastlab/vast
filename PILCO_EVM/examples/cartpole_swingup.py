import numpy as np
import gym
import cartpole_swingup_envs
from pilco.models import PILCO
from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
from gpflow.utilities import print_summary
import os
import random

SEED = 0

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from utils import rollout

tf.config.set_visible_devices([], 'GPU')  # disable GPU

########################################################

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

# T = 30  # the number of timesteps in each random rollout
# # T = 130  # the number of timesteps in each random rollout
# T_sim = T  # the number of timesteps in each rollout that uses the controller
# J = 4  # the number of random rollouts at the beginning before first optimization starts
# N = 3  # the number of rollouts after first optimization (at this stage optimization is performed after each rollout)
# restarts = 1 # the number of times that optimizations with different initializations are performed at each optimization step

#T = 50  # the number of timesteps in each random rollout
T = 50  # the number of timesteps in each random rollout
T_sim = T  # the number of timesteps in each rollout that uses the controller
J = 30  # the number of random rollouts at the beginning before first optimization starts
N = 3  # the number of rollouts after first optimization (at this stage optimization is performed after each rollout)
restarts = 1 # the number of times that optimizations with different initializations are performed at each optimization step
print(J, N)

env = gym.make('CartPole-v1')
env.seed(SEED)
env.action_space.seed(SEED)

########################################################

# Initial random rollouts to generate a dataset
X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=False, render=False)
for i in range(1,J):
    X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=False, render=False)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)




# fix the likelihood variance parameters of the GP models for numerical stability
for model in pilco.mgpr.models:
    model.likelihood.variance.assign(0.001) # 0.001
    set_trainable(model.likelihood.variance, False)

# policy and model optimization
r_new = np.zeros((T, 1))
for rollouts in range(N):
    print("**** ITERATION no", rollouts, " ****")
    pilco.optimize_models(maxiter=maxiter, restarts=restarts)
    pilco.optimize_policy(maxiter=maxiter, restarts=restarts)

    # is_render = False if rollouts == N-1 else True
    X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=False, SUBS=SUBS, render=False)

    # Since we had decide on the various parameters of the reward function
    # we might want to verify that it behaves as expected by inspection
    for i in range(len(X_new)):
        r_new[:, 0] = R.compute_reward(X_new[i,None,:-1], 0.001 * np.eye(state_dim))[0]
    total_r = sum(r_new)
    _, _, r = pilco.predict(X_new[0,None,:-1], 0.001 * S_init, T)
    print("Total ", total_r, " Predicted: ", r)

    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))


########################################################

print('TESTING')
# Test for 1000 timesteps
#input('ENTER TO RUN')


totalSteps = 0
extracted = []
for i in range(100):
    # from IPython import embed; embed()
    _, _, _, _, states_and_dec  = rollout(env, pilco, timesteps=1000, verbose=False, SUBS=SUBS, render=False, extract=True)
    extracted.append(states_and_dec)

# print("Average over 100 trials: ", sum/100)
print('ASDFASD')

#print_summary(pilco)


np.save('100_pole_02', extracted)
# from IPython import embed; embed()

checkpoint = tf.train.Checkpoint(pilco=pilco)
manager = tf.train.CheckpointManager(checkpoint, './100_pole_02', max_to_keep=1)
#tf.saved_model.save(pilco.mgpr, './models')
manager.save(1)


#model = tf.saved_model.load('./models')


########################################################