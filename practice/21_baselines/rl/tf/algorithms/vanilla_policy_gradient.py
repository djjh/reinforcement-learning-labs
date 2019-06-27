from __future__ import annotations

import numpy as np
import tensorflow as tf

from rl.core import Episodes


class VanillaPolicyGradient:

    class HyperParameters:

        _learning_rate = None
        _min_steps_per_batch = None

        def get_learning_rate(self):
            return self._learning_rate

        def get_min_steps_per_batch(self):
            return self._min_steps_per_batch

        class Builder:

            def __init__(self):
                self._learning_rate = None
                self._min_steps_per_batch = None

            def learning_rate(self, learning_rate):
                self._learning_rate = learning_rate
                return self

            def min_steps_per_batch(self, min_steps_per_batch):
                self._min_steps_per_batch = min_steps_per_batch
                return self

            def build(self):
                hyperparameters = VanillaPolicyGradient.HyperParameters()
                hyperparameters._learning_rate = self._learning_rate
                hyperparameters._min_steps_per_batch = self._min_steps_per_batch
                return hyperparameters

        def builder():
            return VanillaPolicyGradient.HyperParameters.Builder()

    class Builder:

        def __init__(self):
            self._environment = None
            self._random_seed = None
            self._policy_factory = None
            self._Rollout = None
            self._hyperparameters = None

        def environment(self, environment):
            self._environment = environment
            return self

        def random_seed(self, random_seed):
            self._random_seed = random_seed
            return self

        def policy_factory(self, policy_factory):
            self._policy_factory = policy_factory
            return self

        def Rollout(self, Rollout):
            self._Rollout = Rollout
            return self

        def min_steps_per_batch(self, min_steps_per_batch):
            self._min_steps_per_batch = min_steps_per_batch
            return self

        def hyperparameters(self, hyperparameters):
            self._hyperparameters = hyperparameters
            return self

        def build(self):
            return VanillaPolicyGradient(
                environment=self._environment,
                random_seed=self._random_seed,
                policy_factory=self._policy_factory,
                Rollout=self._Rollout,
                min_steps_per_batch=self._min_steps_per_batch,
                hyperparameters=self._hyperparameters)

    def builder():
        return VanillaPolicyGradient.Builder()


    def __init__(self, environment, random_seed, policy_factory, Rollout, hyperparameters):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._Rollout = Rollout
        self._min_steps_per_batch = min_steps_per_batch
        self._hyperparameters = hyperparameters

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space

        self._learning_rate = self._hyperparameters.get_learning_rate()
        self._min_steps_per_batch = self._hyperparameters.get_min_steps_per_batch()

        self._graph = None
        self._session = None
        self._policy = None

        self._graph = tf.Graph()
        with self._graph.as_default():

            tf.set_random_seed(self._random_seed)
            # set random seed here, or somewhere
            self._session = tf.Session(graph=self._graph)
            self._policy = policy_factory.create_policy(
                observation_space=self._observation_space,
                action_space=self._action_space,
                session=self._session)

            self._observations = self._policy.get_observations()
            self._actions = self._policy.get_actions()
            self._weights = tf.placeholder(shape=(None,), dtype=tf.float32)

            self._log_probabilities = self._policy.get_log_probabilities()
            self._psuedo_loss = -tf.reduce_mean(self._weights * self._log_probabilities)
            self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._psuedo_loss)

        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def __enter__(self):
        self._session.__enter__()
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.__exit__(exc_type, exc_val, exc_tb)

    def action(self, observation, deterministic):
        return self._policy.action(observation, deterministic)

    def update(self):
        episodes = self._get_episodes()
        batch_observations = self._get_batch_observations(episodes)
        batch_actions = self._get_batch_actions(episodes)
        batch_weights = self._get_batch_weights(episodes)
        self._session.run(
            fetches=self._train,
            feed_dict={
                self._observations: batch_observations,
                self._actions: batch_actions,
                self._weights: batch_weights})

    def _get_episodes(self):
        episodes = Episodes()
        while episodes.num_steps() < self._min_steps_per_batch:
            episode = self._Rollout(
                environment=self._environment,
                policy=self._policy,
                random_seed=self._random_seed,
                deterministic=False,
                render=False)
            episodes.append(episode)
        return episodes

    def _get_batch_observations(self, episodes):
        return [observation
            for episode in episodes
            for observation in episode.get_observations() ]

    def _get_batch_actions(self, episodes):
        return [action
            for episode in episodes
            for action in episode.get_actions() ]

    def _get_batch_weights(self, episodes):
        return [weight
            for episode in episodes
            for weight in self._get_weights(episode)]

    def _get_weights(self, episode):
        rewards = episode.get_rewards()
        return [sum(rewards)] * len(rewards)
