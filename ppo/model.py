import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPONetwork(nn.Module):
    def __init__(self, args, state_size, num_actions):
        super(PPONetwork, self).__init__() 
        self.args = args

        self.input_size = state_size
        self.output_size = num_actions
        self.hidden_units = self.args.hidden_units
        self.device = self.args.device

        # Common layers
        self.linear1 = nn.Linear(self.input_size, self.hidden_units)
        self.linear2 = nn.Linear(self.hidden_units, self.hidden_units)
        
        # Separate heads for actor and critic
        self.actor_head = nn.Linear(self.hidden_units, self.output_size)
        self.critic_head = nn.Linear(self.hidden_units, 1)

        self.std = nn.Parameter(torch.ones(1, self.output_size))
    
    def forward(self, x, action=None):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        action_scores = self.actor_head(x)
        state_values = self.critic_head(x)

        dist = torch.distributions.Normal(action_scores, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        # return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), state_values
        return action, log_prob, torch.zeros(log_prob.size(0), 1), state_values
