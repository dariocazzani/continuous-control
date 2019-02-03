from unityagents import UnityEnvironment
import numpy as np
from tqdm import tqdm

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

env_info = env.reset(train_mode=True)[brain_name]

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
    
    # action = np.zeros(_NUM_ACTIONS)
    predictions = []
    for idx, w in enumerate(agents_weights):
        prediction = np.matmul(np.squeeze(observation[idx]), w) + agents_bias[idx]
        prediction = np.tanh(prediction)
        predictions.append(prediction)
    # prediction = np.matmul(np.squeeze(observation), weights) + bias
    return predictions

if __name__ == "__main__":
    params = np.load('best_params.npy')
    NUM_TRIALS = 4
    total_reward = np.zeros(num_agents)
    for _ in tqdm(range(NUM_TRIALS)):
        env_info = env.reset(train_mode=True)[brain_name]
        observation = env_info.vector_observations
        while True:
            action = decide_actions(observation, [params])
            action = [action]*num_agents
            
            env_info = env.step(np.squeeze(np.asarray(action)))[brain_name] 
            observation = env_info.vector_observations
            reward = np.asarray(env_info.rewards)
            dones = np.asarray(env_info.local_done)

            total_reward += reward
            
            if np.any(dones):
                break
    print("Total reward: {}".format(total_reward/NUM_TRIALS))