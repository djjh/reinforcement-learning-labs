import gym
import numpy as np
import tensorflow as tf


########################
# Configure Parameters #
########################

environment_name = 'CartPole-v0'
num_epochs = 50
min_steps_per_epoch = 5000
policy_learning_rate = 1e-2
action_value_learning_rate = 1e-2
action_value_iterations = 50


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
observation_dimension = np.prod(environment.observation_space.shape)
num_actions = environment.action_space.n


#####################
# Initialize Policy #
#####################

# Initialize placeholders.
observation_placeholder = tf.placeholder(shape=(None, observation_dimension), dtype=tf.float32)
action_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network to learn a policy.
input_layer = observation_placeholder
hidden_layers = [32]
previous_layer = input_layer
for hidden_layer in hidden_layers:
    previous_layer = tf.layers.Dense(units=hidden_layer, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=num_actions, activation=None)(previous_layer)
logits = output_layer

# Build the action sampling operation.
action_operation = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)

# Build policy psuedo loss function.
action_masks = tf.one_hot(indices=action_placeholder, depth=num_actions)
log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits=logits), axis=1)
policy_loss = -tf.reduce_mean(weights_placeholder * log_probabilities)

# Build the policy training operation.
policy_training_operation = tf.train.AdamOptimizer(learning_rate=policy_learning_rate).minimize(policy_loss)


####################################
# Initialize Action Value Function #
####################################

# Initialize additional placeholders not required by the policy.
return_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network to learn the return.
input_layer = tf.concat(values=[observation_placeholder, tf.cast(tf.expand_dims(action_placeholder, 1), dtype=tf.float32)], axis=1)
hidden_layers = [32]
previous_layer = input_layer
for hidden_layer in hidden_layers:
    previous_layer = tf.layers.Dense(units=hidden_layer, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=1, activation=None)(previous_layer)

# Build the on-policy action value function operation.
action_value_operation = tf.squeeze(output_layer, axis=1)

# Build the loss function for learning the action value function.
action_value_loss = tf.reduce_mean((return_placeholder - action_value_operation)**2)

# Build the action value function training operation.
action_value_training_operation = tf.train.AdamOptimizer(learning_rate=action_value_learning_rate).minimize(action_value_loss)


######################
# Initialize Session #
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

    # Initialize lists for policy traning data.
    batch_observations = []
    batch_actions = []
    batch_weights = []
    episode_rewards = []

    # Initialize additional lists for action value function training data.
    batch_returns = []
    episode_observations = []
    episode_actions = []

    # Initialize lists for diagnostic data.
    batch_episode_returns = []
    batch_episode_lengths = []

    # Initialize the environment.
    observation = environment.reset()

    while True:

        # Sample an action from the policy.
        action = session.run(
            action_operation,
            feed_dict={
                observation_placeholder: np.array(observation.reshape(1,-1))
            })[0]

        # Record training data.
        batch_observations.append(observation)
        batch_actions.append(action)
        episode_observations.append(observation)
        episode_actions.append(action)

        # Step the environment by taking the selected action.
        observation, reward, done, info = environment.step(action)

        # Record more training data.
        episode_rewards.append(reward)

        if done:

            # Calculate the cost to go for each step in the episode.
            episode_returns = np.cumsum(episode_rewards[::-1])[::-1].tolist()

            # Record training data.
            batch_returns += episode_returns

            # Record diagnostic data.
            batch_episode_returns.append(episode_returns[0])
            batch_episode_lengths.append(len(episode_returns))

            # Get action value function estimate.
            action_values = session.run(
                action_value_operation,
                feed_dict={
                    observation_placeholder: np.array(episode_observations),
                    action_placeholder: np.array(episode_actions)
                })

            # Record the weights.
            batch_weights += action_values.tolist()

            if len(batch_observations) > min_steps_per_epoch:
                break;

            # Re-initialize episode lists.
            episode_rewards = []
            episode_observations = []
            episode_actions = []

            # Re-initialize the environment.
            observation = environment.reset()


    #################
    # Update Policy #
    #################

    batch_policy_loss, _ = session.run([
            policy_loss,
            policy_training_operation
        ],
        feed_dict={
            observation_placeholder: np.array(batch_observations),
            action_placeholder: np.array(batch_actions),
            weights_placeholder: np.array(batch_weights)
        })


    ################################
    # Update Action Value Function #
    ################################

    for iteration in range(action_value_iterations):
        batch_action_value_loss, _ = session.run([
                action_value_loss,
                action_value_training_operation
            ],
            feed_dict={
                observation_placeholder: np.array(batch_observations),
                action_placeholder: np.array(batch_actions),
                return_placeholder: np.array(batch_returns)
            })


    #######################
    # Display Diagnostics #
    #######################

    print("epoch: %3d\tpolicy loss: %.3f\taction value loss: %.3f\treturn: %.3f\tlength: %.3f" %
        (epoch, batch_policy_loss, batch_action_value_loss, np.mean(batch_episode_returns), np.mean(batch_episode_lengths)))
