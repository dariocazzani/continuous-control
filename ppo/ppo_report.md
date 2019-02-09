# Report

## 1. Training Code

* ### The repository includes functional, well-documented, and organized code for training the agent. <br>
     * [x] **Function `main` at line `189` creates a policy network with randomly initialized weights:**

   ``` python
   policy = PPONetwork(args, _STATE_SIZE, _NUM_ACTIONS)
   ```
     * [x] **Function `train` at line `134` initializes the gradient based optimizer `Adam` and starts the train loop**
   ```python
   def train(policy, args):
    optimizer = optim.Adam(policy.parameters(), args.learning_rate)
    
    episode = 1
    current_rewards = []
    try:
        while True: # Train until requirements are satisfied
        ...
   ```
     * [x] **Function `run_rollout` at line `115` runs a single game for each agent in parallel and saves:**
         * states
         * values
         * actions
         * action lob probabilities
         * rewards
         * terminal state

        **for each agent at each step**
      
     * [x] **Function `process_rollout` at line `97` computes the advantages for each rollout**
     * [x] **Function `ppo_update` at line `69` performs multiple gradient descent steps on the network based on the loss computed with PPO algorithm. Loss is computed between line `83` and `89`**
     
  ``` python
   _, log_probs, entropy_loss, values = policy(states[b], actions[b])
   ratio = (log_probs - log_probs_old[b]).exp() # pnew / pold
   surr1 = ratio * advantages[b]
   surr2 = torch.clamp(surr1, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantages[b]
   policy_surr = -torch.min(surr1, surr2).mean() - (entropy_loss.to(args.device) * args.entropy_coefficent).mean()
   value_loss = 0.5 * (returns[b] - values).pow(2.).mean()
    ```
   
     * [x] **Finally at line `147` we play a round with the current policy and compute the total average for all agents.**
     
     
     
     
     Final avg score for the 20 agents during training: Episode: 61 Current score: 37.94
