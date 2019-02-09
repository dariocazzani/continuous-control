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
    
    The advantage function is answering the question: <br>_How much better was the action that I took based on the expectation of what would normally happen in the state that I am currently in?_
    <br><br>
    
    By multiplying the log probability of the policy actions with the advantage function, we get the final optimization objecting that is used in policy gradient. (See point 6 in the following table)
   
    ![Vanilla](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/vanilla%20policy%20gradient.svg)
    
  * ### Problem and a possible solution:
    If we keep running the gradient descent on 1 batch of collected experience, we would en up updating the parameters of the Neural Net so far out of the range from where the data was collected, that the advantage function (which is already an estimate), is going to be completely wrong.<br>
    
    **TRPO**: To solve this issue one sucessfull approach is to tmake sure that if you are updating the policy you are never going to move too far away from the old policy. <br>
    This idea is at the core of TRPO, which is where **PPO** takes inspiration from. <br>
    To make sure that the updated policy doesn't move too far away from the current policy, TRPO ads a `KL` constraint to the optimization objective.<br>
    This `KL` constraint adds some overhead and can sometimes lead to very undesirable trainig behavior.
    
  * ### PPO:
    Let's first define:
    ![r_theta](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/r_theta.png)
    
    which is the ratio between the new updated policy outputs and the outputs of the previous olf version of the policy network. <br>
    So fiven a sequence of actions and states, the value if `r_theta` will be larger than 1 if the action is more likely now than it was in the old version of the policy, and less than 1 otherwise.<br>
    Then if we multiply `r_theta` with the advantage function, we get the normal TRPO objecting in a more readable form.
    
  * ### The Main Objective Function:
    ![l_clip](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/L_clip.png)
    
    This is an expectation. so we are going to compute it over batches of trajectories. <br>
    This expectation operator is taken over the minimum of 2 terms: `r_theta` and `r_theta, 1-eps, 1+eps`.
    
    Let's understand the objective function by looking at the plot of the surrogate function `l_clip` function as a function of the probability ratio `r_theta`, for positive advantages (left) and negative advantages (right).
    
    ![surrogate_plot](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/surrogate_plot.png)
    
    On the left plot we see actions that yielded better than expected return, while on the righs side we see actions that yielded worse than expected return. <br>
    On the left side the plot flattens out when `r` gets too high and this happens when the actions is a lot more likely under the current policy than it was under the old policy. <br>
    In this case we don't want to overdo the action update too much and so the objective funtion gets clipped to limit the effect of the gradient update. <br>
    Similarly, but with opposite signs, for the right side of the plot. <br>
    
    In other words: **The PPO Objective Function** does the same as the TRPO objective because it forces the policy updates to be conservative if they move very far away from the current policy.
    
  * ### The Algorithm End-to-End:
    ![algo_ppo](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/algo_ppo.png)
    
    There are 2 alternating threads in PPO:
      * With the first one (line `2` to `4`) the current policy is interacting with the environment generating episodes sequences for which we immediately calculate the advantage function using the fitted baseline estimate for the state values.
      * Every so many episodes a second thread (line `6-7`) is going to collect all the experience amd run gradient descent on the policy network using the clipped PPO objective.
      
  * ### The Final Training Objective for PPO:
    ![ppo_objective](https://github.com/dariocazzani/continuous-control/blob/master/ppo/images/ppo_objective.png)
    
    It's the sum of 3 terms:
      * The clipped loss that we just discussed
      * The second term os in charge of updating the baseline network. It's the part of the network that is in charge of estimating how good it is to be in this state, or more specifically what is the average amount of discounted reward that we expect to get from this point onward
      * The last term is called **entropy term**. This term is in charge of making sure that the agent does enough exploration during training. <br>
      In contrast to discrete action policies (that output the action choice probabilities), the PPO policy head outputs the parameters of a **Gaussian Distribution for each available action**.<br>
      When running the agent in trainnig mode, the policy will sample from these distributions to get a continuous output value for each action.<br><br>
      _Why does the entropy encourage exploration?_<br><br>
      **The entropy of a stochastic variable, (which is driven by an underlying probability distribution) is the average amount of bits that is needed to represent the outcome.**<br><br>
      It is a measure of how unpredictable an outcome of the variable is. So maximizing its entropy will force it to have a wide spread over all the possible options resulting in the most unpredictable outcome.
    <br>
    
    Finally, parameters `c1` and `c2` weigh the contributions of these different parts of the cost function
      
       
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
