import numpy as np
import tensorflow as tf

from rl.core import Policy

class VanillaPolicy(Policy):

    def __init__(self, environment, weights, action_sampler, log_directory, random_seed, learning_rate):
        self.environment = environment
        self.weights = weights
        self.action_sampler = action_sampler

        self.log_directory = log_directory
        self.random_seed = random_seed
        self.learning_rate = learning_rate

        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.num_actions = self.environment.action_space.n

        self.graph = tf.Graph()
        update_summaries = []

        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            with tf.variable_scope('Observations'):
                self.observations_placeholder = tf.placeholder(shape=(None, self.observation_dimensions), dtype=tf.float32)

            with tf.variable_scope('Actions'):
                self.actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

            with tf.variable_scope('Weights'):
                self.weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)

            with tf.variable_scope('Policy'):
                layers = [
                    tf.layers.Dense(units=32, activation=tf.tanh),
                    tf.layers.Dense(units=self.num_actions, activation=None)]
                previous_layer = self.observations_placeholder
                for layer in layers:
                    previous_layer = layer(previous_layer)
                logits = previous_layer

            with tf.variable_scope('Sample'):
                self.probabilities = tf.nn.softmax(logits)

            with tf.variable_scope('Loss'):
                self.action_masks = tf.one_hot(indices=self.actions_placeholder, depth=self.num_actions)
                self.log_probabilities = tf.reduce_sum(self.action_masks * tf.nn.log_softmax(logits), axis=1)
                self.psuedo_loss = -tf.reduce_mean(self.weights_placeholder * self.log_probabilities)
                update_summaries.append(tf.summary.scalar('PolicyLoss', self.psuedo_loss))

            with tf.variable_scope('Optimizer'):
                self.training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.psuedo_loss)

            self.update_summary = tf.summary.merge(inputs=update_summaries)

        self.session = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(logdir=self.log_directory + "/policy", graph=self.graph)

    def __enter__(self):
        self.session.__enter__()
        self.writer.__enter__()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.__exit__(exc_type, exc_val, exc_tb)
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def get_parameters(self):
        # self.all_variables = tf.concat(values=[], axis=0)
        return self.session.run(
            fetches=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            feed_dict={})

    def set_parameters(self, parameters):
        # for variable, parameter in zip(self.get_parameters(), parameters):
        #     variable.assign(parameter)
        pass
    #

    def __call__(self, observation):
        probabilities = self.session.run(
            fetches=self.probabilities,
            feed_dict={self.observations_placeholder: np.array(observation.reshape(1, -1))})[0]
        return self.action_sampler.sample(probabilities)

    def update(self, epoch, episodes):
        _, summary = self.session.run(
            fetches=[self.training_operation, self.update_summary],
            feed_dict={
                self.observations_placeholder: self.get_batch_observations(episodes),
                self.actions_placeholder: self.get_batch_actions(episodes),
                self.weights_placeholder: self.weights.get_batch_weights(episodes)})
        self.writer.add_summary(summary, epoch)

    def get_batch_observations(self, episodes):
        return [observation
            for episode in episodes
            for observation in episode.observations ]

    def get_batch_actions(self, episodes):
        return [action
            for episode in episodes
            for action in episode.actions ]
