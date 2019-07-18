# A vanilla policy gradient implementation of a stochastic policy which improves
# the probablities of actions that will result in the most future rewards. The
# implementation assigns credit (in the form of weights) to actions using the
# TD residual; weights = r_t + V(s_t+1) - V(s_t). r_t is a reward not a return

import gym
import numpy as np
import tensorflow as tf


###################################
# Initialize Algorithm Parameters #
###################################

environment_name = 'CartPole-v0'
num_epochs = 50
min_steps_per_epoch = 5000
policy_learning_rate = 1e-2
value_learning_rate = 1e-2
value_training_iterations = 50


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
observation_dimension = np.prod(environment.observation_space.shape)
num_actions = environment.action_space.n


#####################
# Initialize Policy #
#####################

# Initialize placeholders for policy inputs and outputs.
observations_placeholder = tf.placeholder(shape=(None, observation_dimension), dtype=tf.float32)
actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network to generate the un-normalized probability
# distribution over actions for a given state/observation.
input_layer = observations_placeholder
hidden_layer_sizes = [32]
previous_layer = input_layer
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.Dense(units=hidden_layer_size, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=num_actions, activation=None)(previous_layer)
logits = output_layer

# Build the operation for sampling a action from the policy.
action_operation = tf.squeeze(input=tf.random.categorical(logits=logits, num_samples=1), axis=1)

# Build the psuedo loss function for estimating the policy gradient.
action_masks = tf.one_hot(indices=actions_placeholder, depth=num_actions)
log_probabilites = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits=logits), axis=1)
policy_loss = -tf.reduce_mean(weights_placeholder * log_probabilites)

# Build the policy training operation.
policy_training_operation = tf.train.AdamOptimizer(learning_rate=policy_learning_rate).minimize(policy_loss)


###################################
# Initialize State Value Function #
###################################

# Initialize additional placeholders for state value function inputs and outputs
returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build feedforward network to compute state value function given a state/observation.
input_layer = observations_placeholder
hidden_layer_sizes = [32]
previous_layer = input_layer
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.Dense(units=hidden_layer_size, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=1, activation=None)(previous_layer)

# Build the predicting operation for the state value function.
value_operation = tf.squeeze(input=output_layer, axis=1)

# Build the loss function for training the state value function.
value_loss = tf.reduce_mean((returns_placeholder - value_operation)**2)

# Build the training operation for the state value function.
value_training_operation = tf.train.AdamOptimizer(learning_rate=value_learning_rate).minimize(value_loss)


######################
# Initialize Compute #
######################

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


################
# Train Policy #
################

for epoch in range(num_epochs):

    ##########################
    # Generate Training Data #
    ##########################

    # Initialize lists for policy training data.
    batch_observations = []
    batch_actions = []
    batch_weights = []
    episode_rewards = []

    # Initialize lists for state value function training data.
    batch_returns = []
    episode_observations = []

    # Initialize lists for diagnostic information.
    batch_episode_lengths = []
    batch_episode_returns = []

    # Initialize the environment.
    observation = environment.reset()

    while True:

        # Sample an action from the policy.
        action = session.run(
            fetches=action_operation,
            feed_dict={observations_placeholder: np.array(observation.reshape(1,-1))})[0]

        # Record training data.
        batch_observations.append(observation)
        batch_actions.append(action)
        episode_observations.append(observation)

        # Step the environment take the action.
        observation, reward, done, info = environment.step(action)

        # Record training data.
        episode_rewards.append(reward)

        if done:

            # Compute rewards to go.
            episode_returns = np.cumsum(episode_rewards[::-1])[::-1].tolist()

            # Record the state value function training data.
            batch_returns += episode_returns

            # Record diagnostic information.
            batch_episode_lengths.append(len(episode_returns))
            batch_episode_returns.append(episode_returns[0])

            # Compute the values for each state in the episode.
            values = session.run(
                fetches=value_operation,
                feed_dict={observations_placeholder: np.array(episode_observations)})

            # Compute the TD residual: r_t + V(s_t+1) - V(s_t).
            td_residuals = []
            for t in range(len(episode_rewards)):
                r_t = episode_rewards[t]
                V_t = values[t]
                V_t_plus_1 = values[t+1] if t+1 < len(values) else 0
                td_residuals.append(r_t + V_t_plus_1 - V_t)

            # Record the TD residual as the weights.
            batch_weights += td_residuals

            if len(batch_observations) > min_steps_per_epoch:
                break

            # Re-initialize episode lists.
            episode_observations = []
            episode_rewards = []

            # Re-initialize the environment.
            observation = environment.reset()


    #################
    # Update Policy #
    #################

    batch_policy_loss, _ = session.run(
        fetches=[
            policy_loss,
            policy_training_operation],
        feed_dict={
            observations_placeholder: np.array(batch_observations),
            actions_placeholder: np.array(batch_actions),
            weights_placeholder: np.array(batch_weights)})


    #########################
    # Update Value Function #
    #########################

    for iteraction in range(value_training_iterations):
        batch_value_loss, _ = session.run(
            fetches=[
                value_loss,
                value_training_operation],
            feed_dict={
                observations_placeholder: np.array(batch_observations),
                returns_placeholder: np.array(batch_returns)})


    #######################
    # Display Diagnostics #
    #######################

    average_episode_length = np.mean(batch_episode_lengths)
    print("epoch: %3d, value loss per step: %.3f, avg ep return: %.3f avg ep len: %.3f" %
        (epoch, batch_value_loss/average_episode_length, np.mean(batch_episode_returns), average_episode_length))
