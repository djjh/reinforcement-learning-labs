#
# Demonstrates implementing VPG for environments with continuous action
# spaces. This currently get stuck in a local minimum when attempting to solve
# MountainCarContinuous-v0. This is due to the fact that the agent is penalized
# for energy spent and is only given a larger reward for reaching the goal.
# Thus, without some additional exploration technique, a vanilla policy gradient
# will never be able to reach the goal before optmizing for energy use.
#
# Envronment
#     Test Environment: MountainCarContinuous-v0
#     Observation Space: Continuous - Box(1,)
#     Action Space: Continuous - Box(2,)
#
# Policy
#     Policy Update: Vanilla Policy Gradient (aka REINFORCE)
#     Policy Representation: Diagnonal Gaussion Policy with covarience matrix
#                            being represented as a standalone vector of log
#                            standard deviations.
#
#
# References
#     Diagnonal Gaussion Policy - https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
#
#

import gym
import numpy as np
import tensorflow as tf

###################################
# Initialize Algorithm Parameters #
###################################

environment_name = 'MountainCarContinuous-v0'
num_epochs = 50
min_steps_per_epoch = 5000
policy_learning_rate = 1e-2
discount = 0.9
epochs_to_render = (0,25,num_epochs-1)


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
observation_dimensions = sum(environment.observation_space.shape)
action_dimensions = sum(environment.action_space.shape)

assert isinstance(environment.observation_space, gym.spaces.Box), "Observation space must be a Box."
assert isinstance(environment.action_space, gym.spaces.Box), "Action space must be a Box."


#####################
# Initialize Policy #
#####################

# Initialize placeholder and variables for policy input and output.
observations_placeholder = tf.placeholder(name="observations_placeholder", shape=(None, observation_dimensions), dtype=tf.float32)
actions_placeholder = tf.placeholder(name="actions_placeholder", shape=(None, action_dimensions), dtype=tf.float32)
weights_placeholder = tf.placeholder(name="weights_placeholder", shape=(None,), dtype=tf.float32)
# We use log of standard deviations because logs the range becomes (-inf,inf) rather than (0,inf)
# and it's easier to train parameters without contstraints. We need to exponentiate them to
# to get the standard deviations (obviously).
action_log_standard_deviation = tf.get_variable(
    name='action_log_standard_deviation',
    initializer=-0.5*np.ones(action_dimensions, dtype=np.float32))

# Build a feedforward network to generate the mean action given an observation.
input_layer = observations_placeholder
hidden_layer_sizes = [32]
previous_layer = input_layer
for hidden_layer_size in hidden_layer_sizes:
    previous_layer = tf.layers.Dense(units=hidden_layer_size, activation=tf.tanh)(previous_layer)
output_layer = tf.layers.Dense(units=action_dimensions, activation=None)(previous_layer)  # TODO: Should we use an activation?
action_mean = output_layer  # tf.convert_to_tensor(value=output_layer)

# Build an operation to sample an action from the policy given an observation.
# Alternatively, we could also use tf.distributions
action_standard_deviation = tf.math.exp(action_log_standard_deviation)
action_operation = action_mean + tf.random.normal(tf.shape(action_mean)) * action_standard_deviation

# Build the psudeo loss function to obtain the policy gradient and train the policy.
epsilon = 1e-8
action_log_likelihood = -0.5 * tf.reduce_sum(
    ((actions_placeholder - action_mean) / (action_standard_deviation + epsilon))**2
    + 2 * action_log_standard_deviation
    + 2.0*np.log(np.pi)
    , axis=1)
policy_loss = -tf.reduce_mean(weights_placeholder * action_log_likelihood)

# Build an operation to train the policy.
policy_training_operation = tf.train.AdamOptimizer(learning_rate=policy_learning_rate).minimize(policy_loss)


######################
# Initialize Compute #
######################

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


################
# Train Policy #
################

for epoch in range(num_epochs):

    ######################
    # Collect Experience #
    ######################

    # Initialize lists for policy training data.
    batch_observations = []
    batch_actions = []
    batch_weights = []
    episode_rewards = []

    # Initialize lists for diagnostic data.
    batch_episode_lengths = []
    batch_episode_returns = []

    # Initialize environment.
    observation = environment.reset()

    while True:

        if epoch in epochs_to_render:
            if len(batch_episode_returns) == 0:
                environment.render()

        # Sample an action from the policy.
        action = session.run(
            fetches=action_operation,
            feed_dict={observations_placeholder: np.array(observation.reshape(1,-1))})

        # Record training data.
        batch_observations.append(np.squeeze(observation))
        batch_actions.append(np.squeeze(action))

        # Step the environment taking the action.
        observation, reward, done, info = environment.step(action)

        # Record training data.
        episode_rewards.append(reward)

        if done:

            # Compute the discounted reward to go and length.
            episode_returns = list(episode_rewards)
            for t in reversed(range(len(episode_returns)-1)):
                episode_returns[t] += discount * episode_returns[t+1]

            # Record diagnostic info.
            batch_episode_lengths.append(len(episode_rewards))
            batch_episode_returns.append(sum(episode_rewards))

            # Record the reward to go as the weights.
            batch_weights += episode_returns

            # Exit to update policy if collected enough experience.
            if len(batch_observations) > min_steps_per_epoch:
                break

            # Re-initialize pre-episode lists.
            episode_rewards = []

            # Re-initialize the environment.
            observation = environment.reset()


    #################
    # Update Policy #
    #################

    # print(batch_observations.shape)
    batch_policy_loss, _ = session.run(
        fetches=[policy_loss, policy_training_operation],
        feed_dict={
            observations_placeholder: np.array(batch_observations),
            actions_placeholder: np.expand_dims(np.array(batch_actions), axis=1),
            weights_placeholder: np.array(batch_weights)})


    #######################
    # Display Diagnostics #
    #######################

    print("epoch %3d\tloss %.3f\treturn %.3f\tlength %.3f" %
        (epoch, batch_policy_loss, np.mean(batch_episode_returns), np.mean(batch_episode_lengths)))
