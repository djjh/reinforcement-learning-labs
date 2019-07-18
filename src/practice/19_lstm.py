import gym
import io
import numpy as np
import multiprocessing as mp
import os
import tensorflow as tf
import random

from collections import deque
from gym import wrappers
from os.path import splitext, basename
from matplotlib import pyplot as pp
from time import strftime, gmtime


def negative_log(x):
    return -np.log(x)

#################
# Configuration #
#################

experiment_name = splitext(basename(__file__))[0]
timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
log_directory = "log/{}/{}".format(experiment_name, timestamp)
render = True
render_epoch = 49

# environment_name = 'CartPole-v0'
environment_name = 'MountainCar-v0'
random_seed = 0

policy_learning_rate = 1e-3
num_epochs = 50
min_steps_per_epoch = 5000
class PolicyConfig:
    determinisitic = True

density_learning_rate = 1e-3
density_iterations = 100
bonus_coefficient = 1e-0
bonus_function = negative_log
use_actions = False
bonus_only = True

###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)
# environment = wrappers.Monitor(environment, "./gym-results", force=True)

observation_dimensions = np.prod(environment.observation_space.shape)
x_low = environment.observation_space.low
x_high = environment.observation_space.high
num_actions = environment.action_space.n

########################
# Initialize Utilities #
########################

def to_png(figure):
    buffer = io.BytesIO()
    pp.savefig(buffer, format='png')
    pp.close(figure)
    buffer.seek(0)
    return buffer.getvalue()


def histogram(x):
    figure, axis = pp.subplots()
    pp.hist2d(x[:,0], x[:,1], bins=200, range=[[x_low[0], x_high[0]], [x_low[1], x_high[1]]])
    pp.colorbar()
    return figure

def weighted_histogram(x, rb):
    figure, axis = pp.subplots()
    pp.hist2d(x[:,0], x[:,1], bins=200, weights=rb, range=[[x_low[0], x_high[0]], [x_low[1], x_high[1]]])
    pp.colorbar()
    return figure

def probability_contours(points, probabilities):
    figure, axis = pp.subplots()
    # x = points[::100,0]
    # y = points[:100,1]

    z = probabilities.reshape(100, -1).T
    z = bonus_coefficient * bonus_function(z)
    pp.contourf(x, y, z)
    pp.colorbar()

    return figure


class RandomBuffer:

    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def extend(self, values):
        self.buffer.extend(values)
        if len(self.buffer) > self.capacity:
            self.buffer = random.choices(self.buffer, k=self.capacity)

    def sample(self, count=1):
        return random.choices(self.buffer, k=count)

    def size(self):
        return len(self.buffer)


class RingBuffer:
    def __init__(self, capacity):
        self.deque = deque(maxlen=capacity)

    def extend(self, values):
        self.deque.extend(values)

    def sample(self, count=1):
        return [self.deque[i] for i in range(count)]

    def size(self):
        return len(self.deque)


def wrap(layers, previous_layer):
    for layer in layers:
        previous_layer = layer(previous_layer)
    return previous_layer


def argchoice(weights):
    return np.random.choice(range(len(weights)), p=weights)

#####################
# Initialize Policy #
#####################

class Policy:

    def __init__(self, observation_dimensions, num_actions, log_directory, random_seed, learning_rate):
        self.observation_dimensions = observation_dimensions
        self.num_actions = num_actions
        self.log_directory = log_directory
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.graph = self.build_graph()
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

    def build_graph(self):
        self.graph = tf.Graph()
        update_summaries = []
        log_summaries = []
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            with tf.variable_scope('Observations'):
                self.observations_placeholder = tf.placeholder(shape=(None, self.observation_dimensions), dtype=tf.float32)
                for i, observation_dimension in enumerate(tf.split(self.observations_placeholder, self.observation_dimensions, axis=1)):
                    update_summaries.append(tf.summary.histogram('Observations{}'.format(i), observation_dimension))
                    update_summaries.append(tf.summary.scalar('Observations{}Max'.format(i), tf.reduce_max(observation_dimension)))
                    update_summaries.append(tf.summary.scalar('Observations{}Min'.format(i), tf.reduce_min(observation_dimension)))
                    update_summaries.append(tf.summary.scalar('Observations{}Mean'.format(i), tf.reduce_mean(observation_dimension)))
            with tf.variable_scope('Actions'):
                self.actions_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
                update_summaries.append(tf.summary.histogram('Actions', self.actions_placeholder))
            with tf.variable_scope('Weights'):
                self.weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
                update_summaries.append(tf.summary.histogram('Weights', self.weights_placeholder))
            with tf.variable_scope('Diagnostics'):
                self.batch_episode_returns_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
                self.batch_episode_lengths_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
                log_summaries.append(tf.summary.scalar('MaxEpisodeReturn', tf.reduce_max(self.batch_episode_returns_placeholder)))
                log_summaries.append(tf.summary.scalar('MinEpisodeReturn', tf.reduce_min(self.batch_episode_returns_placeholder)))
                log_summaries.append(tf.summary.scalar('MeanEpisodeReturn', tf.reduce_mean(self.batch_episode_returns_placeholder)))
                log_summaries.append(tf.summary.histogram('EpisodeReturns', self.batch_episode_returns_placeholder))
                log_summaries.append(tf.summary.scalar('MeanEpisodeLength', tf.reduce_mean(self.batch_episode_lengths_placeholder)))
                log_summaries.append(tf.summary.scalar('MinEpisodeLength', tf.reduce_min(self.batch_episode_lengths_placeholder)))
                log_summaries.append(tf.summary.histogram('EpisodeLengths', self.batch_episode_lengths_placeholder))
            with tf.variable_scope('Policy'):
                layers = [
                    tf.layers.Dense(units=32, activation=tf.tanh),
                    tf.layers.Dense(units=self.num_actions, activation=None)]
                previous_layer = self.observations_placeholder
                for layer in layers:
                    previous_layer = layer(previous_layer)
                logits = previous_layer
            with tf.variable_scope('Sample'):
                self.probabilites = tf.nn.softmax(logits)
                # self.action_operation = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)
            with tf.variable_scope('Loss'):
                self.action_masks = tf.one_hot(indices=self.actions_placeholder, depth=self.num_actions)
                self.log_probabilities = tf.reduce_sum(self.action_masks * tf.nn.log_softmax(logits), axis=1)
                self.psuedo_loss = -tf.reduce_mean(self.weights_placeholder * self.log_probabilities)
                update_summaries.append(tf.summary.scalar('PsuedoLoss', self.psuedo_loss))
            with tf.variable_scope('Optimizer'):
                self.training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.psuedo_loss)
            self.update_summary = tf.summary.merge(inputs=update_summaries)
            self.log_summary = tf.summary.merge(inputs=log_summaries)
        return self.graph

    def get_action(self, observation):
        probabilites = self.session.run(
            fetches=self.probabilites,
            feed_dict={self.observations_placeholder: np.array(observation.reshape(1, -1))})[0]
        if PolicyConfig.determinisitic:
            return np.argmax(probabilites)
        else:
            return argchoice(probabilites)

    def update(self, epoch, batch_observations, batch_actions, batch_weights):
        _, summary = self.session.run(
            fetches=[self.training_operation, self.update_summary],
            feed_dict={
                self.observations_placeholder: np.array(batch_observations),
                self.actions_placeholder: np.array(batch_actions),
                self.weights_placeholder: np.array(batch_weights),
                self.batch_episode_returns_placeholder: np.array(batch_episode_returns),
                self.batch_episode_lengths_placeholder: np.array(batch_episode_lengths)})
        self.writer.add_summary(summary, epoch)

    def log(self, epoch, batch_episode_returns, batch_episode_lengths):
        summary = self.session.run(
            fetches=self.log_summary,
            feed_dict={
                self.batch_episode_returns_placeholder: np.array(batch_episode_returns),
                self.batch_episode_lengths_placeholder: np.array(batch_episode_lengths)})
        self.writer.add_summary(summary, epoch)

class ProbabilityDensityFunction:

    def __init__(self, dimensions, log_directory, random_seed, learning_rate):
        self.dimensions = dimensions
        self.log_directory = log_directory
        self.random_seed = random_seed
        self.learning_rate = learning_rate

        self.encoder_dimensions = 12
        self.feature_dimensions = 12
        self.model_dimensions = 12

        self.graph = self.build_graph()
        self.session = tf.Session(graph=self.graph)
        self.estimate_writer = tf.summary.FileWriter(logdir=self.log_directory + "/density/estimate", graph=self.graph)
        self.update_writer = tf.summary.FileWriter(logdir=self.log_directory + "/density/update", graph=self.graph)

    def __enter__(self):
        self.session.__enter__()
        self.estimate_writer.__enter__()
        self.update_writer.__enter__()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update_writer.__exit__(exc_type, exc_val, exc_tb)
        self.estimate_writer.__exit__(exc_type, exc_val, exc_tb)
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def build_graph(self):
        self.graph = tf.Graph()
        update_summaries = []
        estimate_summaries = []
        log_summaries = []
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            with tf.variable_scope('X'):
                self.x_placeholder = tf.placeholder(shape=(None, self.dimensions), dtype=tf.float32)
                for i, x_dimension in enumerate(tf.split(self.x_placeholder, self.dimensions, axis=1)):
                    update_summaries.append(tf.summary.histogram('X{}'.format(i), x_dimension))
                self.x = self.x_placeholder
            with tf.variable_scope('Y'):
                self.y_placeholder = tf.placeholder(shape=(None, self.dimensions), dtype=tf.float32)
                # observations0, observations1 = tf.split(self.y_placeholder, [1, 1], 1)
                for i, y_dimension in enumerate(tf.split(self.y_placeholder, self.dimensions, axis=1)):
                    update_summaries.append(tf.summary.histogram('Y{}'.format(i), y_dimension))
                self.y = self.y_placeholder
            with tf.variable_scope('XLabels'):
                x_labels = tf.fill(value=1, dims=(tf.shape(self.x)[0], 1))
            with tf.variable_scope('YLabels'):
                y_labels = tf.fill(value=0, dims=(tf.shape(self.y)[0], 1))
            with tf.variable_scope('Encoder'):
                def relu(output_dimensions):
                    init = 1.0 / np.sqrt(self.encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=tf.nn.relu)
                def linear(output_dimensions):
                    init = 1.0 / np.sqrt(self.encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=None)
                encoder_layers = [relu(self.encoder_dimensions), relu(self.encoder_dimensions), linear(self.feature_dimensions)]
                x_mean = wrap(encoder_layers, self.x)
                y_mean = wrap(encoder_layers, self.y)
                x_features = x_mean
                y_features = y_mean
            with tf.variable_scope('Model'):
                def tanh(output_dimensions):
                    init = 1.0/np.sqrt(self.feature_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=tf.tanh)
                model_layers = [tanh(self.model_dimensions), tanh(1)]
                x_logits = wrap(model_layers, tf.concat(values=[x_features, x_features], axis=1))
                y_logits = wrap(model_layers, tf.concat(values=[x_features, y_features], axis=1))
            with tf.variable_scope('Loss'):
                x_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=x_labels, logits=x_logits)
                labels = tf.concat(values=[x_labels, y_labels], axis=1)
                logits = tf.concat(values=[x_logits, y_logits], axis=1)
                self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
                update_summaries.append(tf.summary.scalar('XLoss', x_loss))
                update_summaries.append(tf.summary.scalar('Loss', self.loss))
            with tf.variable_scope('Estimate'):
                d = tf.nn.sigmoid(x_logits)
                x_probabilities = (1 - d) / d
                self.estimate_operation = x_probabilities
                estimate_summaries.append(tf.summary.histogram('XProbabilities', x_probabilities))
            with tf.variable_scope('Optimizer'):
                self.training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            with tf.variable_scope('Rewards'):
                self.actual_distribution_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.actual_distribution_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Actual', distribution_image)
                log_summaries.append(distribution_image_summary)

                self.weights_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.weights_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Weights', distribution_image)
                log_summaries.append(distribution_image_summary)

                self.bonuses_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.bonuses_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Bonuses', distribution_image)
                log_summaries.append(distribution_image_summary)

            # with tf.variable_scope('Density'):
            #     self.predicted_distribution_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
            #     distribution_image = tf.image.decode_png(contents=self.predicted_distribution_png_placeholder, channels=4)
            #     distribution_image = tf.expand_dims(distribution_image, 0)
            #     distribution_image_summary = tf.summary.image('Predicted', distribution_image)
            #     log_summaries.append(distribution_image_summary)

            self.estimate_summary = tf.summary.merge(inputs=estimate_summaries)
            self.update_summary = tf.summary.merge(inputs=update_summaries)
            self.log_summary = tf.summary.merge(inputs=log_summaries)
        return self.graph

    def estimate(self, epoch, x):
        probabilities, summary = self.session.run(
            fetches=[self.estimate_operation, self.estimate_summary],
            feed_dict={
                self.x_placeholder: np.array(x)})
        self.estimate_writer.add_summary(summary, epoch)
        return np.squeeze(probabilities, axis=1)

    def update(self, epoch, x, y, last):
        _, loss, summary = self.session.run(
            fetches=[self.training_operation, self.loss, self.update_summary],
            feed_dict={
                self.x_placeholder: np.array(x),
                self.y_placeholder: np.array(y)})
        if last:
            self.update_writer.add_summary(summary, epoch)
        return loss

    def log(self, epoch, x, w, rb, grid):
        # probabilities = self.estimate(epoch, grid)
        summary = self.session.run(
            fetches=self.log_summary,
            feed_dict={
                self.actual_distribution_png_placeholder: to_png(histogram(x)),
                # self.predicted_distribution_png_placeholder: to_png(probability_contours(grid, probabilities)),
                self.weights_png_placeholder: to_png(weighted_histogram(x, w)),
                self.bonuses_png_placeholder: to_png(weighted_histogram(x, rb))})
        self.update_writer.add_summary(summary, epoch)


policy = Policy(
    observation_dimensions=observation_dimensions,
    num_actions=num_actions,
    log_directory=log_directory,
    random_seed=random_seed,
    learning_rate=policy_learning_rate)


def get_pdf_input(observations, actions):
    if use_actions:
        o = np.array(batch_observations)
        a = np.expand_dims(np.array(batch_actions), 1)
        oa = np.concatenate((o, a), axis=1)
        return oa.tolist()
    return observations

def get_pdf_domain_dimensions():
    action_dimensions = 1
    if use_actions:
        return observation_dimensions + action_dimensions
    return observation_dimensions

probability_density_function = ProbabilityDensityFunction(
    dimensions=get_pdf_domain_dimensions(),
    log_directory=log_directory,
    random_seed=random_seed,
    learning_rate=density_learning_rate)

# Random seems to work better for the density estimates.
replay_buffer = RandomBuffer(capacity=10*min_steps_per_epoch)
# replay_buffer = RingBuffer(capacity=10*min_steps_per_epoch)





with policy, probability_density_function:

    ################
    # Train Policy #
    ################


    for epoch in range(num_epochs):

        ########################
        # Gather Training Data #
        ########################

        # Initialize lists to hold training data.
        batch_observations = []
        batch_actions = []
        batch_weights = []

        # Initialize lists for diagnostic information.
        batch_episode_returns = []
        batch_episode_lengths = []
        batch_bonuses = []

        # Initialize lists for episodic information.
        episode_observations = []
        episode_actions = []
        episode_rewards = []

        # Initialize the environment and retrieve initial observation.
        observation = environment.reset()

        while True:
            if render and epoch == render_epoch and len(batch_episode_lengths) == 0:
                environment.render()

            action = policy.get_action(observation=observation)

            batch_observations.append(observation)
            batch_actions.append(action)
            episode_observations.append(observation)
            episode_actions.append(action)

            observation, reward, done, info = environment.step(action)

            episode_rewards.append(reward)

            if done:
                x = get_pdf_input(observations=episode_observations, actions=episode_actions)
                density = probability_density_function.estimate(
                    epoch=epoch,
                    x=x)
                bonuses = bonus_coefficient * bonus_function(density)
                if bonus_only:
                    episode_rewards = bonuses
                else:
                    episode_rewards = (np.array(episode_rewards) + bonuses).tolist()

                batch_weights += np.cumsum(episode_rewards[::-1])[::-1].tolist()
                batch_bonuses += bonuses.tolist()
                batch_episode_returns.append(sum(episode_rewards))
                batch_episode_lengths.append(len(episode_rewards))

                if len(batch_observations) >= min_steps_per_epoch:
                    break

                episode_observations = []
                episode_actions = []
                episode_rewards = []

                observation = environment.reset()

        #################
        # Update Policy #
        #################

        policy.update(
            epoch=epoch,
            batch_observations=batch_observations,
            batch_actions=batch_actions,
            batch_weights=batch_weights)

        policy.log(
            epoch=epoch,
            batch_episode_returns=batch_episode_returns,
            batch_episode_lengths=batch_episode_lengths)


        x = get_pdf_input(observations=batch_observations, actions=batch_actions)

        if replay_buffer.size() >= len(x):
            for i in range(density_iterations):
                probability_density_function.update(
                    epoch=epoch,
                    x=x,
                    y=replay_buffer.sample(count=len(x)),
                    last=i==(density_iterations-1))

            # X = np.linspace(start=x_low[0], stop=x_high[0], num=100)
            # :100j, x_low[1]:x_high[1]:100j, 0:1:1j]
            #
            # positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            probability_density_function.log(
                epoch=epoch,
                x=np.array(x),
                w=np.array(batch_weights),
                rb=np.array(batch_bonuses),
                grid=np.array(1)) #positions))
        replay_buffer.extend(list(x))


        print('epoch:', epoch)
# environment.monitor.close()
