import gym
import numpy as np
from common import run_episode

def evaluate(env, get_theta, n_episodes=10**4, max_reward=200):
	"""
	:param env: gym environment
	:param get_theta: random weight generating function (eg. gaussian or uniform)
	:param n_episodes: how many times should we keep searching for max
	:param max_reward: Target reward
	:return: Number of iterations after which max_reward has been achieved for the first time.
	"""
	for i in range(n_episodes):
		theta = get_theta()
		reward = run_episode(env, theta, render=False)
		if reward == max_reward:  # maximum reward in cartpole = 200. Env stops after reaching that
			return i
	return None


def main():
	env = gym.make('CartPole-v0')
	uniform_scores = []
	normal_scores = []

	uniform = lambda: np.random.random(4) * 2 - 1
	normal = lambda: np.random.randn(4) * 2 - 1

	for _ in range(100):
		uniform_scores.append(evaluate(env, uniform))
		normal_scores.append(evaluate(env, normal))
	env.close()

	print('Avg. (out of 100) number of episodes after which return = 200 has been achieved for randomly generated weights:')
	print('UNIFORM distribution: {}'.format(np.mean(uniform_scores)))
	print('NORMAL distribution: {}'.format(np.mean(normal_scores)))


if __name__ == '__main__':
	main()