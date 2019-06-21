import nevergrad as ng
import numpy as np
import tensorflow as tf

from rl.types import Episode, Episodes
from rl.core import Algorithm, Policy, PolicyFactory
from rl.weights import RewardToGoWeights
from gym.spaces import Box, Discrete
from sklearn.preprocessing import PolynomialFeatures


#########################
# Tensorflow Algorithms #
#########################

class DiscreteLinearTensorflowPolicyGradient(Algorithm, Policy):

    def __init__(
        self,
        environment,
        rollout_function,
        min_steps_per_batch):

        self.environment = environment
        self.rollout_function = rollout_function
        self.min_steps_per_batch = min_steps_per_batch

        self.weights_helper = RewardToGoWeights(discount=0.1)

        self.random_seed = 0

        # self.learning_rate = 1e-3
        # self.linear = False
        self.learning_rate = 1e-5
        self.linear = True

        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed);
            self.observations = tf.placeholder(shape=(None, self.observation_dimensions), dtype=tf.float32)
            self.actions = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.weights = tf.placeholder(shape=(None,), dtype=tf.float32)

            if self.linear:
                self.logits = tf.layers.Dense(units=self.action_dimensions, activation=None, use_bias=False)(self.observations)
            else:
                layers = [
                    tf.layers.Dense(units=32, activation=tf.tanh),
                    tf.layers.Dense(units=self.action_dimensions, activation=None)]
                previous_layer = self.observations
                for layer in layers:
                    previous_layer = layer(previous_layer)
                self.logits = previous_layer

            self.probabilities = tf.nn.softmax(self.logits)

            self.action_masks = tf.one_hot(indices=self.actions, depth=self.action_dimensions)
            self.log_probabilities = tf.reduce_sum(self.action_masks * tf.nn.log_softmax(self.logits), axis=1)
            self.psuedo_loss = -tf.reduce_mean(self.weights * self.log_probabilities)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.psuedo_loss)
        self.session = tf.Session(graph=self.graph)

        self.deterministic = False

    def __enter__(self):
        self.session.__enter__()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def get_parameters(self):
        return self.session.run(
            fetches=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            feed_dict={})

    def set_parameters(self, parameters):
        # for variable, parameter in zip(self.get_parameters(), parameters):
        #     variable.assign(parameter)
        pass

    def __call__(self, observation):
        probabilities = self.session.run(
            fetches=self.probabilities,
            feed_dict={self.observations: np.array(observation.reshape(1, -1))})[0]
        if self.deterministic:
            return np.argmax(probabilities)
        else:
            normalized_probabilities = np.array(probabilities) / probabilities.sum()
            return np.random.choice(self.action_dimensions, p=normalized_probabilities)

    def get_policy(self):
        return self

    def update(self):
        self.deterministic = False
        episodes = Episodes()
        while episodes.num_steps() < self.min_steps_per_batch:
            episode = self.rollout_function(self.environment, self, render=False)
            episodes.append(episode)
        _, psuedo_loss = self.session.run(
            fetches=[self.optimize, self.psuedo_loss],
            feed_dict={
                self.observations: episodes.get_batch_observations(),
                self.actions: episodes.get_batch_actions(),
                self.weights: self.weights_helper.get_batch_weights(episodes)})
        self.deterministic = True
