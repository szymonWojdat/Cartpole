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

I decided to compare the average number of episodes needed to achieve score 200 between these two distributions over the course of 100 runs, where one run means running Cartpole episodes until the best score is achieved. Results:
* Normal distribution: 53.06; mean = 0, standard deviation = 1
* Uniform distribution: 11.95; mean = 0, range = 1

**Interpretation:** the expected number of episodes needed in order to reach best score is smaller for the uniform distribution as it is equally likely to select any weight vector. In this particular environemnt the "good" weights must be lying in the area which is less likely picked by the normal distribution.
