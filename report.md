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
     
* ### The repository includes functional, well-documented, and organized code for training the agent.
   * [x] **Check**
   
* ### The submission includes the saved model weights of the successful agent. 
   * [x] **The model is saved in `ppo/ppo.pt`**
   
-----------------------------


## 2. Learning Algorithm

blahblah


## 3. Plot of Rewards

   * **Training**: Training took 61 episodes to learn a policy that would receive an average score of **37.94** across all 20 agents
   
![Training](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/training.png)
   
   * **Testing**: I then ran 100 episodes using the learned policy to make sure that the average score meets the criteria:
   
![testing](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/test.png)
     

## 4. Ideas for Future Work

* **Hyperparameters**: I hand tuned each hyperparameter. I found that even a small change on the batch size or the learning rate would result in the policy not to converge to good results.

    As future work I'd like to experiment with some `Black Box Optimization` algorithms for fine-tuning the hyperparameters and hopefully understand if there exist a set of them that would provide good results even with small pertubations in their values

* **Network size and a different learning algorithm**: The Neural Network used as the "Agent's brain" is quite big despite the small size of the observations state; roughly **282,000 weights**.

    I decided to experiment with **CMA-ES** to learn a policy directly from the observation state. Because it is a black box optimization algorithm, I could train directly a small linear regression network with a total of only **136 weights**. 
    It showed promising results since it got a maximum **average reward of 23**. 
    Quite remarkable considering that the network is 3 orders of magnitude smaller than the Network found with PPO.

    I'd like to continue looking into the CMA-ES solution and hopefully get to a score that meets the criteria for passing.
    For now I am happy with the PPO and the promising results of CMA-ES.
