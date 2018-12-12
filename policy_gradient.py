import tensorflow as tf


def policy_gradient():
    with tf.variable_scope('policy'):
        params = tf.get_variable('policy_parameters', [4, 2])
        state = tf.placeholder('float', [None, 4])
        actions = tf.placeholder('float', [None, 2])
        advantages = tf.placeholder('float', [None, 1])
        linear = tf.matmul(state, params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer
