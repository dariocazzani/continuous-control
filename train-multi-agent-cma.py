import numpy as np
import time, os, joblib, cma
from tqdm import tqdm

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
_NUM_HIDDEN_NEURONS =  4
_NUM_ACTIONS = brain.vector_action_space_size

num_weights_in = _STATE_SIZE*_NUM_HIDDEN_NEURONS
num_bias_in = _NUM_HIDDEN_NEURONS
num_weights_hidden = _NUM_HIDDEN_NEURONS * _NUM_ACTIONS
num_bias_hidden = _NUM_ACTIONS

_NUM_PARAMS = num_weights_in + num_bias_in + num_weights_hidden + num_bias_hidden

ES_PATH = 'es.bin'

def get_weights_bias(params):
    """
        params: list of lenght "num_agents" of parameters for the "brain"
    """
    agents_weights_in = []
    agents_weights_hidden = []
    agents_bias_in = []
    agents_bias_hidden = []

    for p in params:
        p = list(p)
        weights_in = p[:num_weights_in]
        del p[:num_weights_in]
        weights_in = np.reshape(weights_in, [_STATE_SIZE, _NUM_HIDDEN_NEURONS])
        agents_weights_in.append(weights_in)

        bias_in = p[:num_bias_in]
        del p[:num_bias_in]
        agents_bias_in.append(bias_in)
        
        weights_hidden = p[:num_weights_hidden]
        del p[:num_weights_hidden]
        weights_hidden = np.reshape(weights_hidden, [_NUM_HIDDEN_NEURONS, _NUM_ACTIONS])
        agents_weights_hidden.append(weights_hidden)

        bias_hidden = p[:num_bias_hidden]
        del p[:num_bias_hidden]
        agents_bias_hidden.append(bias_hidden)
        
    return agents_weights_in, agents_bias_in, agents_weights_hidden, agents_bias_hidden

def decide_actions(observation, params):
    agents_weights_in, agents_bias_in, agents_weights_hidden, agents_bias_hidden = get_weights_bias(params)
    
    predictions = []
    for idx, w in enumerate(agents_weights_in):
        hidden_layer = np.tanh(np.matmul(np.squeeze(observation[idx]), w) + agents_bias_in[idx])
        prediction = np.matmul(hidden_layer, agents_weights_hidden[idx]) + agents_bias_hidden[idx]
        prediction = np.tanh(prediction)
        predictions.append(prediction)
    return predictions

# def get_weights_bias(params):
#     """
#         params: list of lenght "num_agents" of parameters for the "brain"
#     """
#     agents_weights = []
#     agents_bias = []
    
#     for p in params:
#         weights = p[:_NUM_PARAMS - _NUM_ACTIONS]
#         bias = p[-_NUM_ACTIONS:]
#         weights = np.reshape(weights, [_STATE_SIZE, _NUM_ACTIONS])
#         agents_weights.append(weights)
#         agents_bias.append(bias)
#     return agents_weights, agents_bias

# def decide_actions(observation, params):
#     agents_weights, agents_bias = get_weights_bias(params)
    
#     predictions = []
#     for idx, w in enumerate(agents_weights):
#         prediction = np.matmul(np.squeeze(observation[idx]), w) + agents_bias[idx]
#         prediction = np.tanh(prediction)
#         predictions.append(prediction)
#     return predictions

def play(params, num_trials, train_mode=True):
    agents_reward = np.zeros(len(params))
    for _ in tqdm(range(num_trials)):
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
    
    # Average agents' rewards across num_trials
    return - (agents_reward / num_trials)

def train():
    if not os.path.exists(ES_PATH):
        es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': num_agents})
    else:
        es = joblib.load(ES_PATH)

    rewards_through_gens = []
    generation = 1
    max_avg_rewards = 0
    try:
        while max_avg_rewards < 32:

            # Run quickly at the beginning
            if max_avg_rewards < 10:
                num_trials = 1
            else:
                num_trials = min(100, int(max_avg_rewards * 4))

            solutions = es.ask()
            rewards = play(solutions, num_trials)
            
            es.tell(solutions, rewards)

            rewards = np.array(rewards) *(-1.)
            min_avg_rewards = np.min(rewards)
            max_avg_rewards = np.max(rewards)
            mean_avg_rewards = np.mean(rewards)
            
            if min_avg_rewards > 8:
                num_trials = 50

            if min_avg_rewards > 20:
                num_trials = 100

            print("\n**************")
            print("Generation: {}".format(generation))
            print("Rewards averaged through: {}".format(num_trials))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(min_avg_rewards, max_avg_rewards))
            print("Avg reward: {:.3f}".format(mean_avg_rewards))
            print("**************\n")

            if generation % 50 == 0:
                print("Saving es to {}".format(ES_PATH))
                joblib.dump(es, ES_PATH, compress=1)
                np.save('rewards', rewards_through_gens)

            generation+=1
            rewards_through_gens.append(rewards)


    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return es

if __name__ == '__main__':
    es = train()
    np.save('best_params', es.best.get()[0])