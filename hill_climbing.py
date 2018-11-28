import gym
import numpy as np
from common import run_episode


def evaluate(env, rand_distr, alpha, n_episodes=10**4, max_reward=200, render=False):
	"""
	Evaluates one episode of given env (here most likely cartpole)
	:param env: gym environment
	:param rand_distr: random weight generating distribution (gaussian or uniform)
	:param alpha: learning rate
	:param n_episodes: how many times at most should we keep searching for max
	:param max_reward: Target reward
	:param render: Whether to render the env or not
	:return: Number of iterations after which max_reward has been achieved for the first time.
	"""
	best_reward = None
	if rand_distr == 'uniform':
		best_theta = np.random.random(4) * 2 - 1
		noise = lambda: (np.random.random(4) * 2 - 1) * alpha
	elif rand_distr == 'normal':
		best_theta = np.random.randn(4) * 2 - 1
		noise = lambda: (np.random.randn(4) * 2 - 1) * alpha
	else:
		raise ValueError('rand_distr parameter must be either normal or uniform')

	for i in range(n_episodes):
		theta = best_theta + noise()
		reward = run_episode(env, theta, render=render)
		if best_reward is None or reward > best_reward:
			best_theta = theta
			best_reward = reward
		if reward == max_reward:  # maximum reward in cartpole = 200. Env stops after reaching that
			return i
	else:
		return n_episodes


def run_hill_climbing(learn_rate, num_runs):
	env = gym.make('CartPole-v0')
	uniform_scores = []
	normal_scores = []

	for _ in range(num_runs):
		uniform_scores.append(evaluate(env, 'uniform', learn_rate))
		normal_scores.append(evaluate(env, 'normal', learn_rate))
	env.close()

	print('Avg. (out of {}) number of episodes after which return = 200 has been achieved for randomly generated\
		weights/noise:'.format(num_runs))
	print('UNIFORM distribution: {}'.format(np.mean(uniform_scores)))
	print('NORMAL distribution: {}'.format(np.mean(normal_scores)))

	return {'uniform': uniform_scores, 'normal': normal_scores}
