import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

import gym
import nevergrad as ng
import numpy as np
import random

from rl import *

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

    def action(self, observation, deterministic):
        return self.policy.action(observation, deterministic)

    # Move into the TF Linear Policy.
    # def get_parameters(self):
    #     return self.session.run(
    #         fetches=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
    #         feed_dict={})
    #
    # def set_parameters(self, parameters):
    #     # for variable, parameter in zip(self.get_parameters(), parameters):
    #     #     variable.assign(parameter)
    #     pass

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



class VPG:

    def __init__(self, environment, random_seed, policy_factory, create_rollout, min_steps_per_batch):
        self.environment = environment
        self.random_seed = random_seed
        self.policy_factory = policy_factory
        self.create_rollout = create_rollout
        self.min_steps_per_batch = min_steps_per_batch
        self.deterministic_update_policy = False

        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.policy = self.policy_factory.create(
            observation_space=self.observation_space,
            action_space=self.action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        self.policy_return = -np.inf
        self.policy_steps = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        return self.policy.action(observation, deterministic)

    def update(self):
        episodes = Episodes()
        episodes_probabilities = []

        while episodes.num_steps() < self.min_steps_per_batch:
            recording_policy = RecordingPolicy(self.policy)
            episode = self.create_rollout(
                self.environment,
                recording_policy,
                random_seed=self.random_seed,
                deterministic=self.deterministic_update_policy,
                render=False)
            episodes.append(episode)
            episodes_probabilities.append(recording_policy.probabilities)

        grads = []
        rewards = []
        for i in range(len(episodes)):
            episode_grads = []
            episode_rewards = []
            episode = episodes[i]
            episode_probabilities = episodes_probabilities[i]
            for j in range(len(episode)):
                observation = episode.observations[j]
                action = episode.actions[j]
                reward = episode.rewards[j]
                probabilities = episode_probabilities[j]

                softmax = probabilities
                s = softmax.reshape(-1,1)
                grad_softmax = (np.diagflat(s) - np.dot(s, s.T))
                grad_softmax = grad_softmax[action,:]
                grad_log = grad_softmax / softmax[action]

                episode_grads.append(grad_log[None,:].T.dot(observation[None,:]))
                episode_rewards.append(reward)
            grads.append(episode_grads)
            rewards.append(episode_rewards)

        for i in range(len(grads)):
            for j in range(len(grads[i])):
                self.policy.model += 0.0025 * grads[i][j] * sum([ r * (0.99 ** r) for t,r in enumerate(rewards[i][j:])])

        # print(self.policy.model)

environment_name = 'CartPole-v0'
# environment_name = 'MountainCar-v0'
# environment_name = 'Pendulum-v0'
random_seed = 0
max_epochs = 1000
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

def algorithm_function(environment):
    return VPG(
        environment=environment,
        random_seed=random_seed,
        policy_factory=LinearPolicyFactory(),
        create_rollout=rollout,
        min_steps_per_batch=200)

if __name__ == '__main__':
    run(algorithm_function, environment_function, specification, random_seed, max_epochs)
