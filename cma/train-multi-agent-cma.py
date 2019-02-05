import numpy as np
import time, os, joblib, cma
from tqdm import tqdm
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from actor import Actor

"""
Initialize environments
"""
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

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

_ES_PATH = 'es.bin'
_REWARDS_PATH = 'rewards.bin'
_GENERATION_PATH = 'generations.npy'
_EXPECTED_REWARD = 40
_MIN_NUM_TRIALS = 8

def play(actor, params, num_trials, train_mode=True):
    agents_reward = np.zeros(len(params))
    for _ in tqdm(range(num_trials)):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        observation = env_info.vector_observations
        
        total_rewards = np.zeros(len(params))
        steps = 0
        while True:
            
            action = actor.decide_actions(observation, params)
            
            env_info = env.step(np.asarray(action))[brain_name] 
            observation = env_info.vector_observations
            reward = np.asarray(env_info.rewards)
            dones = np.asarray(env_info.local_done)

            total_rewards += reward
           
            steps += 1
            if np.any(dones):
                break
       
        agents_reward += total_rewards
    
    # Average agents' rewards across num_trials
    return - (agents_reward / num_trials)


def train():
    actor = Actor(_STATE_SIZE, _NUM_ACTIONS)

    if not os.path.exists(_ES_PATH) or not os.path.exists(_GENERATION_PATH) or not os.path.exists(_REWARDS_PATH):
        es = cma.CMAEvolutionStrategy(actor.get_num_params() * [0], 0.1, {'popsize': num_agents})
        rewards_through_gens = []
        generation = 1
    else:
        es = joblib.load(_ES_PATH)
        rewards_through_gens = joblib.load(_REWARDS_PATH)
        generation = np.load(_GENERATION_PATH)

        plt.plot(rewards_through_gens)
        plt.show()


    max_avg_rewards = 0
    best_so_far = 0
    # Linear schedule for num_trials (line passing across (1,1)  and (30,100))
    m = 99./29.
    b = -70./29
    try:
        while max_avg_rewards < _EXPECTED_REWARD:

            # Run quickly at the beginning
            # num_trials = max(_MIN_NUM_TRIALS, int(max_avg_rewards*m + b))
            num_trials = _MIN_NUM_TRIALS

            solutions = es.ask()
            rewards = play(actor, solutions, num_trials)
            
            es.tell(solutions, rewards)

            rewards = np.array(rewards) *(-1.)
            min_avg_rewards = np.min(rewards)
            max_avg_rewards = np.max(rewards)
            mean_avg_rewards = np.mean(rewards)

            print("\n**************")
            print("Generation: {}".format(generation))
            print("Rewards averaged through: {}".format(num_trials))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(min_avg_rewards, max_avg_rewards))
            print("Avg reward: {:.3f}".format(mean_avg_rewards))
            print("**************\n")

            generation+=1
            rewards_through_gens.append(rewards)

            if max_avg_rewards > best_so_far:
                best_so_far = max_avg_rewards
                print("Saving es to {}".format(_ES_PATH))
                joblib.dump(es, _ES_PATH, compress=1)
                joblib.dump(rewards_through_gens, _REWARDS_PATH, compress=1)
                np.save(_GENERATION_PATH, generation)

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return es

if __name__ == '__main__':
    es = train()
    np.save('best_params', es.best.get()[0])