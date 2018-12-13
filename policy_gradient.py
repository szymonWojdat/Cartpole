import tensorflow as tf


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
        newvals = tf.placeholder('float', [None, 1])

        # feedforward - predict the value of given state - using a NN with one hidden layer
        w1 = tf.get_variable('w1', [4, 10])
        b1 = tf.get_variable('b1', [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable('w2', [10, 1])
        b2 = tf.get_variable('b2', [1])
        calculated = tf.matmul(h1, w2) + b2

        # calculating loss
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss
