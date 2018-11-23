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
Work in progress.
