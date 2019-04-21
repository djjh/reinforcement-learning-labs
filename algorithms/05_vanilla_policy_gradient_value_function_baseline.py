import gym
import numpy as np
import tensorflow as tf



########################
# Algorithm Parameters #
########################

environment_name = 'CartPole-v0'
num_epochs = 50
min_steps_per_epoch = 5000
policy_learning_rate = 1e-2
value_learning_rate = 1e-2
value_iterations = 50


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
num_actions = environment.action_space.n
observation_dimension = sum(environment.observation_space.shape)


#####################
# Initialize Policy #
#####################

# Initialize placeholders for observation, action, and weights.
observation_placeholder = tf.placeholder(shape=(None, observation_dimension), dtype=tf.float32)
action_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network that maps observations to an un-normalized
# probability distribution over actions.
input_layer = observation_placeholder
hidden_layers = [32]
previous_layer = input_layer
for hidden_layer in hidden_layers:
    previous_layer = tf.layers.Dense(units=hidden_layer, activation=tf.tanh)(inputs=previous_layer)
output_layer = tf.layers.Dense(units=num_actions, activation=None)(previous_layer)
logits = output_layer

# Build the action sampling operation.
action_operation = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)

# Build the policy's psuedo loss function.
action_masks = tf.one_hot(action_placeholder, num_actions)
log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
policy_loss = -tf.reduce_mean(weights_placeholder * log_probabilities)

# Build the training operation.
policy_training_operation = tf.train.AdamOptimizer(learning_rate=policy_learning_rate).minimize(policy_loss)


#######################################
# Initialize Value Function Estimator #
#######################################

# Initialize the placeholder for returns.
return_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network to approximate the value function.
input_layer = observation_placeholder
hidden_layers = [32]
previous_layer = input_layer
for hidden_layer in hidden_layers:
    previous_layer = tf.layers.Dense(units=hidden_layer, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=1, activation=None)(previous_layer)

# Build the value function estimation operation.
value_operation = tf.squeeze(output_layer, axis=1)

# Build the value function esitmator's loss function, used to approximate the
# cumulative rewards/return that would be recieved for starting in a given state.
# value_loss = tf.losses.mean_squared_error(labels=return_placeholder, predictions=value_operation)
value_loss = tf.reduce_mean((return_placeholder - value_operation)**2)

# Build the value frunction estimation training operation.
value_training_operation = tf.train.AdamOptimizer(learning_rate=value_learning_rate).minimize(value_loss)


################
# Train Policy #
################

# Initialize tensorflow session.
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for epoch in range(num_epochs):

    ########################
    # Gather Training Data #
    ########################

    # Initialize lists for policy training data
    batch_observations = []
    batch_actions = []
    batch_weights = []

    # Initialize lists for additional training data for value function approximator.
    batch_returns = []

    # Initialize lists for diagnostic information
    batch_episode_estimated_returns = []
    batch_episode_returns = []
    batch_episode_lengths = []

    # Initialize list for current episode rewards
    episode_observations = []
    episode_rewards = []

    # Initialize the episode
    observation = environment.reset()

    while True:

        # Sample an action from the policy.
        action = session.run(
            action_operation,
            feed_dict={
                observation_placeholder: np.array(observation.reshape(1,-1))
            })[0]

        # Record the observation and action into training data.
        batch_observations.append(observation)
        batch_actions.append(action)

        # Record the observation into value estimation data.
        episode_observations.append(observation)

        # Step the environment, taking the action.
        observation, reward, done, info = environment.step(action)

        # Record the reward.
        episode_rewards.append(reward)

        if done:
            # Calculate and record the returns into training data.
            episode_returns = np.cumsum(episode_rewards[::-1])[::-1].tolist()
            batch_returns += episode_returns

            # Calculate the value baseline.
            estimated_values = session.run(
                value_operation,
                feed_dict={
                    observation_placeholder: np.array(episode_observations)
                }).tolist()

            # Calculate and record the weights into training data.
            batch_weights += np.subtract(episode_returns, estimated_values).tolist()

            # Record diagnostic info.
            estimated_episode_return = estimated_values[0]
            episode_return = episode_returns[0]
            batch_episode_estimated_returns.append(estimated_episode_return)
            batch_episode_returns.append(episode_return)
            batch_episode_lengths.append(len(episode_rewards))


            # End epoch if collected enough samples.
            if len(batch_observations) > min_steps_per_epoch:
                break;

            # Re-initialize the list for episode rewards.
            episode_rewards = []

            # Re-initialize the list for episode observations.
            episode_observations = []

            # Re-initialize the environment.
            observation = environment.reset()


    #################
    # Update Policy #
    #################

    # Update policy.
    batch_policy_loss, _ = session.run(
        [
            policy_loss,
            policy_training_operation,

        ],
        feed_dict={
            observation_placeholder: np.array(batch_observations),
            action_placeholder: np.array(batch_actions),
            weights_placeholder: np.array(batch_weights)
        })

    # Update value function estimator.
    for i in range(value_iterations):
        batch_value_loss, _ = session.run(
            [
                value_loss,
                value_training_operation,

            ],
            feed_dict={
                observation_placeholder: np.array(batch_observations),
                return_placeholder: np.array(batch_returns)
            })


    ###########################
    # Display Diagnostic Info #
    ###########################

    mean_length = np.mean(batch_episode_lengths)
    print("epoch: %3d\tpolicy loss: %.3f\tvalue loss: %.3f\treturn: %.3f\testimated return: %.3f\tlength: %.3f" %
        (epoch,
        batch_policy_loss/mean_length,
        batch_value_loss/mean_length,
        np.mean(batch_episode_returns),
        np.mean(batch_episode_estimated_returns),
        mean_length,))
