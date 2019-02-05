import numpy as np
import time, os, joblib, cma
from tqdm import tqdm
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from actor import Actor

"""
Initialize environments
"""
env = UnityEnvironment(file_name='_Reacher_Linux/Reacher.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations

_STATE_SIZE = states.shape[1]
_NUM_ACTIONS = brain.vector_action_space_size


if __name__ == "__main__":
    actor = Actor(_STATE_SIZE, _NUM_ACTIONS)

    params = np.load('best_params.npy')
    NUM_TRIALS = 100
    total_reward = np.zeros(num_agents)
    for t in tqdm(range(NUM_TRIALS)):
        env_info = env.reset(train_mode=True)[brain_name]
        observation = env_info.vector_observations
        while True:
            action = actor.decide_actions(observation, [params])
            action = [action]*num_agents
            
            env_info = env.step(np.squeeze(np.asarray(action)))[brain_name] 
            observation = env_info.vector_observations
            reward = np.asarray(env_info.rewards)
            dones = np.asarray(env_info.local_done)

            total_reward += reward
            
            if np.any(dones):
                break
        print("Reward so far: {}".format(total_reward / (t+1)))
    print("Total reward: {}".format(total_reward/NUM_TRIALS))