# from spinningup.spinup.algos.vpg import vpg
from spinup import vpg  # fixme - there's something wrong with this, doesn't import for w/e reason
import tensorflow as tf
import gym


def env_fn(): gym.make('CartPole-v0')


ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=100)
