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

	# TODO - implement hill climbing here

	prev_reward = None
	# get first (completely random) theta here
	# theta = get_theta()
	for i in range(n_episodes):
		reward = run_episode(env, theta, render=False)
		if prev_reward is None or reward > prev_reward:
			pass
		if reward == max_reward:  # maximum reward in cartpole = 200. Env stops after reaching that
			return i
	return None


def main():
	pass


if __name__ == '__main__':
	main()