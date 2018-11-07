import numpy as np

def run_episode(env, theta, render=False):
	"""
	Runs a single episode given env and weight vector.
	"""
	observation = env.reset()
	done = False
	total_reward = 0
	while not done:
		if render: env.render()
		action = np.dot(observation, theta.reshape(-1, 1)).flatten()[0] > 0
		# action = env.action_space.sample()  # take a random action
		observation, reward, done, _ = env.step(action)
		total_reward += reward
	return total_reward