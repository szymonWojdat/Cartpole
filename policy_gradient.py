import tensorflow as tf
import gym
import spinningup as spinup

def main():
	env_fn = lambda: gym.make('CartPole-v0')
	spinup.vpg(env_fn)


if __name__ == '__main__':
	main()
