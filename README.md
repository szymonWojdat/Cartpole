# Cartpole
Cartpole is one of the simplest environments in OpenAI gym. From the docs:
"A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center."
Also worth adding is that an episode will end as soon as the reward (in this case: number of steps without failing) hits 200.
Cartpole is one of the OpenAI Requests for Research which are a collection of "important and fun problems to help new people enter the field". I decided to try solving a couple of them to practice my reinforcement learning skills.
The authors have suggested trying out three relatively straightforward approaches in order to create a successful agent:
