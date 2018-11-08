import gym
import numpy as np
from common import run_episode

def evaluate(env, rand_distr, n_episodes=10**4, max_reward=200):
	"""
	:param env: gym environment
	:param rand_distr: random weight generating distribution (gaussian or uniform)
	:param n_episodes: how many times at most should we keep searching for max
	:param max_reward: Target reward
	:return: Number of iterations after which max_reward has been achieved for the first time.
	"""

	if rand_distr == 'uniform':
		get_theta = lambda: np.random.random(4) * 2 - 1
	elif rand_distr == 'normal':
		get_theta = lambda: np.random.randn(4) * 2 - 1
	else:
		raise ValueError('rand_distr parameter must be either normal or uniform')

	for i in range(n_episodes):
		theta = get_theta()
		reward = run_episode(env, theta, render=False)
		if reward == max_reward:  # maximum reward in cartpole = 200. Env stops after reaching that
			return i
	return n_episodes


def run_random_search():
	env = gym.make('CartPole-v0')
	num_of_runs = 100
	uniform_scores = []
	normal_scores = []

	for _ in range(num_of_runs):
		uniform_scores.append(evaluate(env, 'uniform'))
		normal_scores.append(evaluate(env, 'normal'))
	env.close()

	print('Avg. (out of {}) number of episodes after which return = 200 has been achieved for randomly generated weights:'\
	      .format(num_of_runs))
	print('UNIFORM distribution: {}'.format(np.mean(uniform_scores)))
	print('NORMAL distribution: {}'.format(np.mean(normal_scores)))
