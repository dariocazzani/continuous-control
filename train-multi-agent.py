from unityagents import UnityEnvironment
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

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
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

"""
Brain
"""
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)
        
        # self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        actions = dist.sample()
        return actions, dist, value



def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def test_env():
    model.eval()
    env_info = env.reset(train_mode=True)[brain_name]
    scores = np.zeros(num_agents)
    state = env_info.vector_observations 
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).to(device)
        action, _, _ = model(state)
        # action = dist.sample()
        env_info = env.step(action.cpu().numpy())[brain_name] 
        next_state = env_info.vector_observations         # get next state (for each agent)
        scores += env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done      
        state = next_state
        
    return scores

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(ppo_iters, batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in tqdm(range(ppo_iters)):
        rand_ids = np.random.randint(0, states.size(0), batch_size)
        state = states[rand_ids, :]
        action = actions[rand_ids, :]
        old_log_probs = log_probs[rand_ids, :]
        return_ = returns[rand_ids, :]
        advantage = advantages[rand_ids, :]

        _, dist, value = model(state)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(action)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        actor_loss  = - torch.min(surr1, surr2).mean()
        critic_loss = (return_ - value).pow(2).mean()

        loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_inputs  = states.shape[1]
num_outputs = brain.vector_action_space_size

#Hyper params:
hidden_size      = 400
lr               = 1e-4
batch_size       = 64
ppo_iters        = 30

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

frame_idx  = 0
test_rewards = []

try:
    for _ in range(50):
    # while True:
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        scores = np.zeros(num_agents) 
        while True:
            state = env_info.vector_observations                  # get the current state (for each agent)
            state = torch.FloatTensor(state).to(device)
            action, dist, value = model(state)

            env_info = env.step(action.cpu().numpy())[brain_name] 
            next_state = env_info.vector_observations         # get next state (for each agent)
            # reward = env_info.rewards                         # get reward (for each agent)
            # reward = np.array(reward)
            scores += env_info.rewards
            done = env_info.local_done                        # see if episode finished
            done = np.array(done)

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(scores).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            frame_idx += 1

            if np.any(done):                                  # exit loop if episode finished
                break
        print("Episode done")
    
        # test_reward = np.mean(test_env())
        # test_rewards.append(test_reward)
        # print("test reward: {}".format(test_reward))
        print("Reward: {}".format(torch.cat(rewards).mean()))

        next_state = torch.FloatTensor(next_state).to(device)
        _, _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        ppo_update(ppo_iters, batch_size, states, actions, log_probs, returns, advantage)
except KeyboardInterrupt:
    print("Manual Interrupt")



env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    states = torch.FloatTensor(states).to(device)
    action, dist, value = model(states)
    env_info = env.step(action.cpu().numpy())[brain_name] 
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     print(scores)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break

# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))