import numpy as np
from gpflow import config
from gym import make
import cartpole_swingup_envs
float_type = config.default_float()
import time


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False, sum=False, extract=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        numsteps = 0
        state_and_dec = []
        for timestep in range(timesteps):
            if render: 
                env.render()
                #time.sleep(0.025) # slow down rendering
            u = policy(env, pilco, x, random)
            u = 1 if u > 0 else 0
            if extract == True:
                current = []
                for i in x:
                    current.append(i)
                current.append(u)
                state_and_dec.append(current)
            
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                numsteps +=1
                if done:
                    print("Total steps: ", numsteps)
                    break
                if render: 
                    env.render()
                    #time.sleep(0.025) # slow down rendering
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            if done: break
        if extract is True:
            return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, state_and_dec
        if sum is True:
            return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, numsteps
        else:
            return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()