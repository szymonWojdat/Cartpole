# Cartpole
Cartpole is one of the simplest environments in OpenAI gym. From the docs:

>A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

Also worth adding is that an episode will end as soon as the reward (in this case: number of steps without failing) hits 200.
Cartpole is one of the OpenAI Requests for Research which are a collection of "important and fun problems to help new people enter the field". I decided to try solving a couple of them to practice my reinforcement learning skills.
The authors have suggested trying out three relatively straightforward approaches in order to create a successful agent: random search, hill climbing and policy gradient.

## Random search
The authors have specified this approach in the following way:
>The random guessing algorithm: generate 10,000 random configurations of the model's parameters, and pick the one that achieves the best cumulative reward. It is important to choose the distribution over the parameters correctly.
Focusing on the last sentence, I decided to try out both the normal and uniform distribution.

The math behind behind this approach is actually pretty straightforward - given that the state of Cartpole environment is essentially a four-element vector, one can use a 1x4 vector of weights and calculate their dot product. The sign of this dot product is then used to indicate the direction in which the cartpole will be moved.

I decided to compare the average number of episodes needed to achieve score 200 between these two distributions over the course of 1000 runs, where one run means running Cartpole episodes until the best score is achieved. Moreover, I decided to create a histogram of the results as calculating only the mean can be often misleading.
* Normal distribution: 53.06; mean = 0, standard deviation = 1
* Uniform distribution: 11.95; mean = 0, range = 1

![1](https://github.com/szymonWojdat/Cartpole/blob/master/graphs/random_search_histograms.png)

**Interpretation:** the expected number of episodes needed in order to reach best score is on average smaller for the uniform distribution as it is equally likely to select any weight vector. In this particular environemnt the "good" weights must be lying in the area which is less likely picked by the normal distribution.

## Hill climbing
The authors have specified this approach in the following way:
>The hill-climbing algorithm: Start with a random setting of the parameters, add a small amount of noise to the parameters, and evaluate the new parameter configuration. If it performs better than the old configuration, discard the old configuration and accept the new one. Repeat this process for some number of iterations. How long does it take to achieve perfect performance?

Again, we're using here a 1x4 vector of weights, except this time, the weights get randomly generated only for the first run. In the next iterations, as described above, only a small amount of noise gets added to the weight vector so that it creates a new vector which gets saved only if the performance has improved.

Similarly to random search, I decided to compare the performance between uniform and normal distributions using two performance metrics - average number of runs required to achieve perfect score and histograms. The learning rate (here: signal-to-noise ratio) was set to 0.1.

* Normal distribution: 8140.06; mean = 0, standard deviation = 1
* Uniform distribution: 5684.97; mean = 0, range = 1

![2](https://github.com/szymonWojdat/Cartpole/blob/master/graphs/hill_climbing_histograms.png)

**Interpretation:** the expected number of episodes needed in order to reach best score is on average smaller for the uniform distribution. Hill climbing performs much worse than random search on average, however many times hill climbing was able to find the correct weights pretty fast, especially when the noise was generated uniformly at random. This means that this approach is capable of achieving good results but also likely to get stuck. One solution to this could be dynamically scaling leraning rate - starting with a big one and decreasing it with the number of iterations.

## Policy gradient
The authors have specified this approach in the following way:
>Policy gradient algorithm: here, instead of choosing the action as a deterministic function of the sign of the weighted sum, make it so that action is chosen randomly, but where the distribution over actions (of which there are two) depends on the numerical output of the inner product. Policy gradient prescribes a principled parameter update rule. Your goal is to implement this algorithm for the simple linear model, and see how long it takes to converge.

Policy gradient is a significantly more complicated algorithm compared to hill climbing and random search. Its general idea is to parametrize agent's policy - create a direct mapping from the state of the environment to agent's action. This mapping function is a simple neural network - it takes the state vector and outputs an actions vector. In order to improve the policy little by little, it is necessary to make it stochastic instead of deterministic. What does it mean? In case of random search and hill climbing, we used a hard threshold of 0 - if the product of state and weights was greater than it, the agent always moved right, otherwise it always moved left. Here, instead of making it always perform the same action for a given output, we make it take that action with some probability, which depends on the output of the policy neural network.

But how do we improve the policy?

The only measure of agent's performance is the reward that it gets, so a very natural idea would be to update the parameters in the direction of getting more reward. What does this mean? When we see that the agent gets some reward after taking a step in the environment, we take derivatives of its parameters and update them with the direction of the derivatives - going uphill. The derivatives of the parameters vector are called the gradient, which is why this algorithm is called the policy gradient.

The agent is going to take some actions in the environment and update its policy over time but remember that we need to start somewhere and a fresh agent does not have any knowledge about the environment whatsoever. It means that the agent's early actions will be likely quite bad and certainly not optimal. This means that the variance will be big - it will take a long time for the policy to converge to the optimal policy because simply, from an expert's point of view, the agent will keep taking some "random" actions. In order to change this, we can try to learn the state-value function of the environment which will basically tell us how good it is to be in a certain state. The agent will still keep taking its actions according to its policy parameters rather than the value of the state but we can tweak the value towards which we keep updating the policy parameters. Instead of taking a pure reward, we can subtract the estimated state value from that reward.

If this seems a bit confusing, I can share the way I like to think about it - the value function is updated every step, just like the policy parameters, so it's going to be based on exactly the same information we have about the environemnt as the policy function. So we can say that the value function is a human-readable reflection of the policy - the higher the value of a state is, the larger is the chance that the policy will make the agent end up in this state. So if we "overrated" a given state - the return that comes after going through it will be lower than that state's value according to the value function, then we need to make it less likely for us to step into it again. And it makes sense, since the reward minus state value, towards which we update the policy parameters will be negative, so we'll end up decreasing policy's probability of going through that state again. On the other hand, if the reward minus state value is positive, this means we "underrated" this state, so we want to go up with the gradient and increase the probability of going through it again..

Below you can see a histogram of policy gradient's number of episodes required to solve the environment.

![2](https://github.com/szymonWojdat/Cartpole/blob/master/graphs/policy_gradient_histograms.png)

**Interpretation:** policy gradient turns out to be not as effective as random search in case of Cartpole. As mentioned before, it improves only little by little while the actual solution is not very sophisticated, which is why random search is so effective. Simple methods seem to work best for simple environments.

