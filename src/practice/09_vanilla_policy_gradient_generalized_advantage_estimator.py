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
gae_gamma_discount = 0.92
gae_lambda_discount = 0.92


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
observation_dimension = np.prod(environment.observation_space.shape)
num_actions = environment.action_space.n


#####################
# Initialize Policy #
#####################

# Initialize placeholders for policy inputs/outputs.
observations_placeholder = tf.placeholder(shape=(None, observation_dimension), dtype=tf.float32)
actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedforward network to compute the un-normalized probabiliyt
# distribution over actions given an observation.
input_layer = observations_placeholder
hidden_layer_sizes = [32]
previous_layer = input_layer
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.Dense(units=hidden_layer_size, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=num_actions, activation=None)(previous_layer)
logits = output_layer

# Build an operation for sampling an action from the policy.
action_operation = tf.squeeze(input=tf.random.categorical(logits=logits, num_samples=1), axis=1)

# Build the psuedo loss function for obtaining the policy gradient.
action_masks = tf.one_hot(indices=actions_placeholder, depth=num_actions)
log_probabilites = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits=logits), axis=1)
policy_loss = -tf.reduce_mean(weights_placeholder * log_probabilites)

# Build an operation for training the policy.
policy_training_operation = tf.train.AdamOptimizer(learning_rate=policy_learning_rate).minimize(policy_loss)


#############################
# Initialize Value Function #
#############################

# Initialize additional placeholders for value function inputs/outputs.
returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

# Build a feedfoward network to compute the expected return given an
# observation.
input_layer = observations_placeholder
hidden_layer_sizes = [32]
previous_layer = input_layer
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.Dense(units=hidden_layer_size, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=1, activation=None)(previous_layer)

# Build an operation for estimating the expected return given an observation.
value_operation = tf.squeeze(input=output_layer, axis=1)

# Build the loss function for the value function.
value_loss = tf.reduce_mean((returns_placeholder - value_operation)**2)

# Build a operation to train the value function.
value_training_operation = tf.train.AdamOptimizer(learning_rate=value_learning_rate).minimize(value_loss)


##############################
# Initialize Compute Session #
##############################

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
    episode_observations = []
    episode_actions = []
    episode_rewards = []

    # Initialize lists for value function training data.
    batch_returns = []

    # Initialize lists for diagnostic information.
    batch_episode_lengths = []
    batch_episode_returns = []

    # Obtain the initialize observations.
    observation = environment.reset()

    while True:

        # Sample an action from the policy.
        action = session.run(
            fetches=action_operation,
            feed_dict={observations_placeholder: np.array(observation.reshape(1,-1))})[0]

        # Record diagnostic and training data.
        batch_observations.append(observation)
        batch_actions.append(action)
        episode_observations.append(observation)
        episode_actions.append(action)

        # Step the evironment taking the action.
        observation, reward, done, info = environment.step(action)

        # Record diagnostic and training data.
        episode_rewards.append(reward)

        if done:

            # Compute the episode return.
            episode_returns = np.cumsum(episode_rewards[::-1])[::-1].tolist()

            # Record diagnostic and training information.
            batch_episode_lengths.append(len(episode_returns))
            batch_episode_returns.append(episode_returns[0])
            batch_returns += episode_returns

            # Compute the values for the observations.
            values = session.run(
                fetches=value_operation,
                feed_dict={observations_placeholder: np.array(episode_observations)})

            # Compute advantages using the generalized advantage function.
            td_residuals = []
            for t in range(len(episode_rewards)):
                r_t = episode_rewards[t]
                V_t_plus_1 = values[t+1] if t+1 < len(values) else 0
                V_t = values[t]
                td_residuals.append(r_t + gae_gamma_discount * V_t_plus_1 - V_t)
            advantages = []
            discount = 1
            for t in reversed(range(len(td_residuals))):
                advantage = td_residuals[t]
                advantage += gae_gamma_discount * gae_lambda_discount * td_residuals[t+1] if t+1 < len(td_residuals) else 0
                advantages.append(advantage)

            # Record the weights for training the policy.
            batch_weights += advantages

            if len(batch_observations) > min_steps_per_epoch:
                break;

            # Re-initialize episode lists.
            episode_observations = []
            episode_actions = []
            episode_rewards = []

            # Obtain the initialize observations.
            observation = environment.reset()


    #################
    # Update Policy #
    #################

    batch_policy_loss, _ = session.run(
        fetches=[policy_loss, policy_training_operation],
        feed_dict={
            observations_placeholder: np.array(batch_observations),
            actions_placeholder: np.array(batch_actions),
            weights_placeholder: np.array(batch_weights)})


    #########################
    # Update Value Function #
    #########################

    for iteration in range(value_training_iterations):
        batch_value_loss, _ = session.run(
            fetches=[value_loss, value_training_operation],
            feed_dict={
                observations_placeholder: np.array(batch_observations),
                returns_placeholder: np.array(batch_returns)})


    #######################
    # Display Diagnostics #
    #######################

    print("epoch: %3d\tvloss= %.3f\tploss= %.3f\treturn: %.3f\tlength: %.3f" %
        (epoch, batch_value_loss, batch_policy_loss, np.mean(batch_episode_returns), np.mean(batch_episode_lengths)))
