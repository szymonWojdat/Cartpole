import tensorflow as tf
import random
import gym
import numpy as np


def policy_gradient():
	with tf.variable_scope('policy'):
		'''
		an important thing to keep in mind here is that we might actually update the policy for multiple state/action
		pairs at once, just like in case of batched supervised learning, which is why we're using None so many times.
		We never know how steps will there be in an episode and we're updating at the end for all steps as it's
		based on Monte-Carlo
		'''
		# we need to find params that point to the more valuable action
		params = tf.get_variable('policy_parameters', [4, 2])
		state = tf.placeholder('float', [None, 4])
		action_vec = tf.placeholder('float', [None, 2])  # one hot which says which action was taken
		advantages = tf.placeholder('float', [None, 1])  # this says if we need to increase/decrease probs for actions

		# feedforward - predict - calculate probabilities for actions
		linear = tf.matmul(state, params)
		probabilities = tf.nn.softmax(linear)

		# calculating how wrong was the prediction and in which direction - loss is the "sum of all mistakes"
		prob_of_taken_action = tf.mul(probabilities, action_vec)  # calculated probability of the action that was taken
		good_probabilities = tf.reduce_sum(prob_of_taken_action, reduction_indices=[1])  # sum columns
		eligibility = tf.log(good_probabilities) * advantages  # policy gradient update
		loss = -tf.reduce_sum(eligibility)
		optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
		return probabilities, state, action_vec, advantages, optimizer


def value_gradient():
	with tf.variable_scope('value'):
		# we pass the state for which to update and "actual action values" (based on return)
		state = tf.placeholder('float', [None, 4])
		actual_values = tf.placeholder('float', [None, 1])

		# feedforward - predict the value of given state - using a NN with one hidden layer
		w1 = tf.get_variable('w1', [4, 10])
		b1 = tf.get_variable('b1', [10])
		h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
		w2 = tf.get_variable('w2', [10, 1])
		b2 = tf.get_variable('b2', [1])
		calc_values = tf.matmul(h1, w2) + b2

		# calculating loss
		diffs = calc_values - actual_values
		loss = tf.nn.l2_loss(diffs)
		optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
		return calc_values, state, actual_values, optimizer, loss


def run_episode(env, policy_grad, value_grad, sess, lmbd, render=False):
	p_probabilities, p_state, p_actions, p_advantages, p_optimizer = policy_grad
	v_calc_value, v_state, v_actual_values, v_optimizer, v_loss = value_grad
	observation = env.reset()
	totalreward = 0
	states_memo = []
	actions_memo = []
	advantages_memo = []
	return_vals_memo = []
	transitions_memo = []  # a list of tuples (state, action (num, not vec) taken from it, reward after this action)

	# rollout an episode
	done = False
	while not done:
		if render: env.render()
		# get an action according to the policy
		obs_vector = np.expand_dims(observation, axis=0)
		probs = sess.run(p_probabilities, feed_dict={p_state: obs_vector})
		action = 0 if random.uniform(0, 1) < probs[0][0] else 1

		# remember the state and the action vector
		states_memo.append(observation)
		action_vector = np.zeros(2)
		action_vector[action] = 1
		actions_memo.append(action_vector)

		# take an action
		old_observation = observation
		observation, reward, done, info = env.step(action)
		transitions_memo.append((old_observation, action, reward))
		totalreward += reward

	for transition_num, transition in enumerate(transitions_memo):
		obs, action, reward = transition

		# converting timestep rewards into a discounted MC return
		future_reward = 0
		future_transitions = len(transitions_memo) - transition_num
		current_discount = 1
		for i in range(future_transitions):
			reward_in_timestep_j = transitions_memo[transition_num + i][2]
			future_reward += reward_in_timestep_j * current_discount
			current_discount = current_discount * lmbd

		# get current value estimation
		obs_vector = np.expand_dims(obs, axis=0)
		current_calc_values = sess.run(v_calc_value, feed_dict={v_state: obs_vector})[0][0]

		# advantage and return values - targets for policy and value fn parameter updates
		advantages_memo.append(future_reward - current_calc_values)
		return_vals_memo.append(future_reward)

	update_vals_vector = np.expand_dims(return_vals_memo, axis=1)
	advantages_vector = np.expand_dims(advantages_memo, axis=1)

	# update value function parameters
	sess.run(v_optimizer, feed_dict={v_state: states_memo, v_actual_values: update_vals_vector})
	# this will be needed for graphing
	# real_loss = sess.run(v_loss, feed_dict={v_state: states_memo, v_actual_values: update_vals_vector})

	# update policy parameters
	sess.run(p_optimizer, feed_dict={p_state: states_memo, p_advantages: advantages_vector, p_actions: actions_memo})

	return totalreward


def evaluate(env, n_episodes=10**4, max_reward=200, render=False, discount=0.97):
	policy_grad = policy_gradient()
	value_grad = value_gradient()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for i in range(n_episodes):
		reward = run_episode(env, policy_grad, value_grad, sess, discount, render=render)
		if reward == max_reward:  # maximum reward in cartpole = 200. Env stops after reaching that
			return i
	return n_episodes


def run_policy_gradient(num_runs, render=False):
	env = gym.make('CartPole-v0')
	scores = []

	for _ in range(num_runs):
		scores.append(evaluate(env, render=render))
	env.close()
	print('Policy gradient mean score: {}'.format(np.mean(scores)))

	return scores
