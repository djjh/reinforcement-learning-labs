import gym
import numpy as np
import tensorflow as tf

# Notes:
# - Numpy shapes with `None` dimension means any dimension.
# - Numpy reshape will treat a -1 dimension as long as it needs to be given the
#   other dimensions and the total number of dimensions in the original shape.
# - Pretty sure ndarray.reshape(1,-1) is equal to ndarray.flatten()

environment_name = 'CartPole-v0'
num_epochs = 50
min_steps_per_epoch = 5000
learning_rate = 1e-2

#########################
# Setup the environment #
#########################

environment = gym.make(environment_name)
num_actions = environment.action_space.n
observation_dimension = environment.observation_space.shape[0]


####################
# Setup the policy #
####################

observation_placeholder = tf.placeholder(shape=(None,observation_dimension),dtype=tf.float32)
weights_placeholder = tf.placeholder(shape=(None,),dtype=tf.float32)
action_placeholder = tf.placeholder(shape=(None,),dtype=tf.int32)

# Build a feed-foward network that generates the next action given an state
# (which is approximated by an observation in this case)
previous_layer = observation_placeholder
hidden_layer_sizes = [32]
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.dense(previous_layer, units=hidden_layer_size, activation=tf.tanh)
output_layer = tf.layers.dense(previous_layer, units=num_actions, activation=None)

# Selecting an action based on the state of the output layer from the feedforward MLP above
# is a "Discrete Choice" problem: https://en.wikipedia.org/wiki/Discrete_choice .
# Here, we let tensorflow decide the best such algorithm for our case using tf.random.categorical

# logits are the un-normalized log probabilities for the categories/classes
logits = output_layer

# tf.squeeze is required because categorical returns shape=[batch_size, num_samples]. Since
# we are setting num_samples to 1, and squeezing axis 1 (the axis with num_samples dimensions),
# this will drop that dimension. Thus we are changing the shape to just [batch_size].
# action_operation = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)
action_operation = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

# We need to setup the pseudo loss function, which we will be using to get the policy gradient.

# The action_placeholder encodes a discrete action as an integer in the set [0..num_actions].
# In order to
action_masks = tf.one_hot(indices=action_placeholder, depth=num_actions)
# TODO: describe this
log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
# TODO: describe this
psuedo_loss = -tf.reduce_mean(weights_placeholder * log_probabilities)


# Define the traingin operation, which will optimize the model according to the
# gradient of the psuedo loss function.
train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(psuedo_loss)


session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())




####################
# Train the policy #
####################


for epoch in range(num_epochs):

    # Generate Training Data by rolling out the policy repeatedly until
    # we have enough training samples (min_steps_per_epoch)

    # Initialize some lists to collect the training data for the policy.
    batch_observations = []
    batch_actions = []
    batch_returns = []

    # Initialize a list for collecting episode lengths to provide progress
    # information to the user during training.
    episode_returns = []
    episode_lengths = []

    # Initialize the environment and retrieve the initial observation.
    observation = environment.reset()

    # Initialize a list to collect
    episode_rewards = []

    while True:

        # Generate a trajectory/episode/rollout.

        # Sample an action from the policy.
        action = session.run(
            action_operation,
            feed_dict={
                observation_placeholder: np.array(observation.reshape(1,-1)) # TODO try and replace with flatten()
            })[0]

        # Record the observation and action into the batch training data.
        batch_observations.append(observation)
        batch_actions.append(action)

        # Step the environment by sampling from its transition function
        observation, reward, done, info = environment.step(action)

        # Record the reward for the current episode.
        episode_rewards.append(reward)

        if done:

            # Compute the return for each state as the cumulative reward for the entire
            # trajectory. Ideally, you would only want to sum reward that followed
            # a given state, but for simplicity, we are using the return for the
            # whole trajectory as an approximation.
            episode_return = sum(episode_rewards)
            episode_length = len(episode_rewards)
            batch_returns += [episode_return] * episode_length

            # Record the episode length for debugging.
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            if len(batch_observations) > min_steps_per_epoch:
                break

            # Re-Initialize the environment and retrieve the initial observation.
            observation = environment.reset()

            # Re-Initialize a list to collect
            episode_rewards = []

    # Use the returns as the weights for training the policy. The reason for
    # separating the idea of weight from returns is that there are other values
    # that can serve as alternative weights that will also result in a policy
    # gradient that can be used for training.
    batch_weights = batch_returns

    # Train
    batch_loss, _ = session.run(
        [psuedo_loss, train_operation],
        feed_dict={
            observation_placeholder: np.array(batch_observations),
            action_placeholder: np.array(batch_actions),
            weights_placeholder: np.array(batch_weights)
        })

    # Display Progress

    print("epoch: %3d\tloss: %.3f\treturn: %.3f\tepisode length: %.3f" %
        (epoch, batch_loss, np.mean(episode_returns), np.mean(episode_lengths)))
