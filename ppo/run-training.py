import sys
from collections import deque

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from unityagents import UnityEnvironment

from model import PPONetwork
from utils import Batcher

import argparse
parser = argparse.ArgumentParser(description='PPO for Reacher')
parser.add_argument('--device', type=str, default='cuda',
                    help="Select device for training and inference")
parser.add_argument('--discount-rate', type=float, default=0.99,
                    help='')
parser.add_argument('--tau', type=float, default=0.95,
                    help='')
parser.add_argument('--gradient-clip', type=float, default=5,
                    help='')
parser.add_argument('--rollout-length', type=int, default=2048,
                    help='')
parser.add_argument('--ppo-epochs', type=int, default=10,
                    help='')
parser.add_argument('--ppo-clip', type=float, default=2.0,
                    help='')
parser.add_argument('--batch-size', type=int, default=32,
                    help='')
parser.add_argument('--entropy-coefficent', type=float, default=1E-2,
                    help='')
parser.add_argument('--required-reward', type=float, default=30,
                    help='')
parser.add_argument('--learning-rate', type=float, default=3E-4,
                    help='')
parser.add_argument('--hidden-units', type=int, default=512,
                help='')
args = parser.parse_args()


env = UnityEnvironment(file_name='../Reacher_Linux/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

_STATE_SIZE = env_info.vector_observations.shape[1]
_NUM_ACTIONS = brain.vector_action_space_size
_NUM_AGENTS = len(env_info.agents)

def play(policy, args):
    env_info = env.reset(train_mode=True)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(_NUM_AGENTS)                         
    while True:
        actions, _, _, _ = policy(torch.FloatTensor(states).to(args.device))
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)

def ppo_update(args, policy, optimizer, processed_rollout):
    # Create batch 
    states, actions, log_probs_old, returns, advantages = list(map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout)))
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    memory_size = int(states.shape[0])
    batcher = Batcher(memory_size // args.batch_size, [np.arange(memory_size)])
    for _ in range(args.ppo_epochs):
        batcher.shuffle()
        while not batcher.end():
            b = batcher.next_batch()[0]
            b = torch.Tensor(b).long()
  
            _, log_probs, entropy_loss, values = policy(states[b], actions[b])
            ratio = (log_probs - log_probs_old[b]).exp() # pnew / pold
            surr1 = ratio * advantages[b]
            surr2 = torch.clamp(surr1, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantages[b]
            policy_surr = -torch.min(surr1, surr2).mean() - (entropy_loss.to(args.device) * args.entropy_coefficent).mean()

            value_loss = 0.5 * (returns[b] - values).pow(2.).mean()
            optimizer.zero_grad()
            (policy_surr + value_loss).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5)
            optimizer.step()

    return optimizer, policy

def process_rollout(rollout, last_value):
    processed_rollout = [None] * (len(rollout) - 1)
    advantages = torch.FloatTensor(np.zeros((_NUM_AGENTS, 1))).to(args.device)
    returns = last_value.detach()
    for i in reversed(range(len(rollout) - 1)):
        states, value, actions, log_probs, rewards, terminals = rollout[i]
        terminals = torch.FloatTensor(terminals).unsqueeze(1).to(args.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(args.device)
        actions = actions
        states = torch.FloatTensor(states).to(args.device)
        next_value = rollout[i + 1][1]
        returns = rewards + args.discount_rate * terminals * returns

        td_error = rewards + args.discount_rate * terminals * next_value.detach() - value.detach()
        advantages = advantages * args.tau * args.discount_rate * terminals + td_error
        processed_rollout[i] = [states, actions, log_probs, returns, advantages]
    return processed_rollout

def run_rollout(policy, args):
        rollout = []

        env_info = env.reset(train_mode=True)[brain_name]    
        states = env_info.vector_observations  
        for _ in range(args.rollout_length):
            actions, log_probs, _, values = policy(torch.FloatTensor(states).to(args.device))
            env_info = env.step(actions.cpu().detach().numpy())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
                    
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states
        
        last_value = policy(torch.FloatTensor(states).to(args.device))[-1]
        rollout.append([states, last_value, None, None, None, None])
        return rollout, last_value

def train(policy, args):
    optimizer = optim.Adam(policy.parameters(), args.learning_rate)
    
    episode = 1
    current_rewards = []
    try:
        while True: # Train until requirements are satisfied
            rollout, last_value = run_rollout(policy, args)
        
            processed_rollout = process_rollout(rollout, last_value)

            optimizer, policy = ppo_update(args, policy, optimizer, processed_rollout)
                
            mean_reward = play(policy, args)
            current_rewards.append(mean_reward)
            print('Episode: {} Current score: {:.2f}'.format(episode, mean_reward))
            episode += 1
            
            plt.clf()
            plt.plot(range(len(current_rewards)), current_rewards, label='Rewards')
            plt.xlabel('Episodes', fontsize=18)
            plt.ylabel('Reward', fontsize=18)
            plt.legend(loc='best', shadow=True, fancybox=True)
            plt.pause(0.005)
            if mean_reward >= args.required_reward+5:
                print("Required met. Finishing and saving")
                break
        plt.show()

    except KeyboardInterrupt:
        print("Manual Interrupt")
    except Exception as e:
        print("Something went wrong: {}".format(e))

    torch.save(policy.state_dict(), "ppo.pt")
    return policy
    
def test(policy, args):
    print("Testing...")
    min_episodes = 100
    all_rewards = []
    for _ in tqdm(range(min_episodes)):
        mean_reward = play(policy, args)
        all_rewards.append(mean_reward)
        avg = np.mean(all_rewards)
        print(avg)
    
    plt.plot(range(len(all_rewards)), all_rewards, label='Rewards')
    plt.plot(range(len(all_rewards)), [avg]*len(all_rewards), label='Average')
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)
    plt.show()

if __name__ == "__main__":    
    policy = PPONetwork(args, _STATE_SIZE, _NUM_ACTIONS)
    policy.to(args.device)
    policy = train(policy, args)
    test(policy, args)