import gym
import numpy as np
import tensorflow as tf

def mulitlayer_perceptron(input_layer, layer_sizes, activation, output_activation):
    # Creates a feed forward network and returns the output layer.
    previous_layer = input_layer
    for layer_size in layer_sizes[:-1]:
        previous_layer = tf.layers.dense(
            previous_layer,
            units=layer_size,
            activation=activation)
    return tf.layers.dense(
        previous_layer,
        units=layer_sizes[-1],
        activation=output_activation)


class Episode:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_observation(self, observation):
        self.observations.append(observation)

    def add_action(self, action):
        self.actions.append(action)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def get_return(self):
        return sum(self.rewards)

    def get_length(self):
        return len(self.rewards)

    def get_weights(self):
        return [self.get_return()] * self.get_length()


class Episodes:

    def __init__(self):
        self.episodes = []

    def add_episode(self, episode):
        self.episodes.append(episode)

    def get_num_episodes(self):
        return len(self.episodes)

    def get_observations(self):
        return [observation
            for episode in self.episodes
            for observation in episode.observations]

    def get_actions(self):
        return [action
            for episode in self.episodes
            for action in episode.actions]

    def get_returns(self):
        return [episode.get_return()
            for episode in self.episodes]

    def get_lengths(self):
        return [episode.get_length()
            for episode in self.episodes]

    def get_weights(self):
        return [weight
            for episode in self.episodes
            for weight in episode.get_weights()]

    def get_total_length(self):
        return sum(self.get_lengths())


class Policy:

    def __init__(self, environment, learning_rate):
        # Initialize the policy network.
        self.observation_placeholder = tf.placeholder(shape=(None, environment.observation_space.shape[0]), dtype=tf.float32)
        logits = mulitlayer_perceptron(self.observation_placeholder, [32, environment.action_space.n], tf.tanh, None)

        # Initialize the action selection operation.
        self.action = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

        self.weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.action_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

        action_masks = tf.one_hot(self.action_placeholder, environment.action_space.n)

        log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

        # Initialize the pseudo loss function used to obtain the policy gradient.
        self.psuedo_loss = -tf.reduce_mean(self.weights_placeholder * log_probabilities)

        # Initialize the training operation.
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.psuedo_loss)

        # Initialize the tensorflow session.
        # TODO: check out the other types of sessions.
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def sample(self, observation):
        return self.session.run(
            self.action,
            feed_dict={
                self.observation_placeholder: observation.reshape(1, -1)
            })[0]

    def update(self, episodes):
        batch_loss, _ = self.session.run(
            [self.psuedo_loss, self.train],
            feed_dict={
                self.observation_placeholder: np.array(episodes.get_observations()),
                self.action_placeholder: np.array(episodes.get_actions()),
                self.weights_placeholder: np.array(episodes.get_weights())
            })

        return batch_loss, episodes.get_returns(), episodes.get_lengths()


environment_name = 'CartPole-v0'
num_epochs = 50  # Same as the number of training batches.
min_steps_per_epoch = 5000
learning_rate = 1e-2

# Initialize the environment.
environment = gym.make(environment_name)

# Initialize a policy for the environment.
policy = Policy(environment, learning_rate)

# Training loop.
for epoch in range(num_epochs):

    ###########
    # ROLLOUT #
    ###########

    # Retrieve the initial observation corresponding to the initial state.
    observation = environment.reset()

    # One training batch consists of as many episodes as is required
    # to reach min_steps_per_epoch.
    episodes = Episodes()
    episode = Episode()
    while True:

        # Optionally Rrender the environment for debugging via visual
        # inspection.
        if episodes.get_num_episodes() == 0:
            environment.render()

        # Generate the next actions given the current observation.
        action = policy.sample(observation)

        # Step the environment for the
        next_observation, reward, done, info = environment.step(action)

        # Record the observation, action, and reward for the current step.
        episode.add_observation(observation)
        episode.add_action(action)
        episode.add_reward(reward)

        if done:

            # Add the episode to the batch of episodes
            episodes.add_episode(episode)

            # We are done with the current epoch if we've accumulated the minumum
            # number of steps across all of episodes.
            if episodes.get_total_length() > min_steps_per_epoch:
                break
            else:

                # Reset the environment.
                observation = environment.reset()

                #
                episode = Episode()

        else:
            #
            observation = next_observation

    #########
    # TRAIN #
    #########

    batch_loss, batch_rets, batch_lens = policy.update(episodes)
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (epoch, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
