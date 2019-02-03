import numpy as np
import time 
from tqdm import tqdm

import cma

from unityagents import UnityEnvironment

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
_NUM_PARAMS = _NUM_ACTIONS * _STATE_SIZE + _NUM_ACTIONS

def get_weights_bias(params):
    """
        params: list of lenght "num_agents" of parameters for the "brain"
    """
    agents_weights = []
    agents_bias = []
    
    for p in params:
        weights = p[:_NUM_PARAMS - _NUM_ACTIONS]
        bias = p[-_NUM_ACTIONS:]
        weights = np.reshape(weights, [_STATE_SIZE, _NUM_ACTIONS])
        agents_weights.append(weights)
        agents_bias.append(bias)
    return agents_weights, agents_bias

def decide_actions(observation, params):
    agents_weights, agents_bias = get_weights_bias(params)
    
    predictions = []
    for idx, w in enumerate(agents_weights):
        prediction = np.matmul(np.squeeze(observation[idx]), w) + agents_bias[idx]
        prediction = np.tanh(prediction)
        predictions.append(prediction)
    return predictions

def play(params, train_mode=True):
    _NUM_TRIALS = 12
    agents_reward = np.zeros(len(params))
    for _ in tqdm(range(_NUM_TRIALS)):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        observation = env_info.vector_observations
        
        total_rewards = np.zeros(len(params))
        steps = 0
        while True:
            
            action = decide_actions(observation, params)
            
            env_info = env.step(np.asarray(action))[brain_name] 
            observation = env_info.vector_observations
            reward = np.asarray(env_info.rewards)
            dones = np.asarray(env_info.local_done)

            total_rewards += reward
           
            steps += 1
            if np.any(dones):
                break
       
        agents_reward += total_rewards
    
    # Average agents' rewards across NUM_TRIALS
    return - (agents_reward / _NUM_TRIALS)

def train():
    es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': num_agents})
    rewards_through_gens = []
    generation = 1
    try:
        while not es.stop():
            solutions = es.ask()
            rewards = play(solutions)
            
            es.tell(solutions, rewards)

            rewards = np.array(rewards) *(-1.)
            print("\n**************")
            print("Generation: {}".format(generation))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            print("Avg reward: {:.3f}".format(np.mean(rewards)))
            print("**************\n")

            generation+=1
            rewards_through_gens.append(rewards)
            np.save('rewards', rewards_through_gens)

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return es

if __name__ == '__main__':
    es = train()
    np.save('best_params', es.best.get()[0])
    input("Press enter to play... ")
    score = play(es.best.get(), train_mode=False)
print("Final Score: {}".format(-score))