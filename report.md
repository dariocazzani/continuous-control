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

  * ### Intro:
    If we are doing supervised learning, we can have a static dataset. In this way we can be relative sure that aftert running a SGD optimizer on that data, we will converge to a decent local optima. <br>
    The problem with RL is that the training data that is generated is itself dependent on the current policy because the agent is generating its own training data by interacting with the environment rather than relying on a static dataset.<br>
    The data distributions over the observations and rewards are constanlty changin as the agent learns, and this is a cause for instability. <br>
    Furthermore RL is very sensitive to hyperparameters tuning and initialization.<br>
    To address this problems, a team at OpenAI designe an algorithm called **Proximal Policy Optimization**
    
  * ### Details:
    **PPO** is a policy gradient method. This means that unilke other popular algorithms like DQN (that can learn from offline stored data), PPO learn online. <br>
  It doesn't use a replay buffer to store past experiences, but instead it learns directly from what the agent encounters in the environment. <br>
  Once the experience has been used to do a gradient update, the experience is discarded.
  
  * ### Vanilla Policy Gradient Algorithms:
    A general polocu optimization method usually starts by defining a policy gradient loss as the expectation over the log of the policy actions, times an estimate of the advantage function.
    * The agent's policy is a neural network that takes the observations from the environment as an input and suggests actions to take as output.
    * The advantage function tries to estimate what the relative velu of the selected action is in the current state
    
  * ### Advantage function:
    In order to copute the advantage function we need:
    
       * Discounted sum of rewards: it is a weighted sum of all the rewards the agent got at each time step during the current episode.
       * Baseline or value function: it tries to give an estimate of the discounted sum of rewards from the current state onwards. <br>
        Because this output is the output of a neural net, it's going to be a noisy estimate --> high variance.
    
    `Advangage function = discounted rewards -  baseline estimate`
    <br>
    The advantage function is answering the question: _How much better was the action that I took based on the expectation of what would normally happen in the state that I am currently in?_
    <br>
    By multiplying the log probability of the policy actions with the advantage function, we get the final optimization objecting that is used in policy gradient
   


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
