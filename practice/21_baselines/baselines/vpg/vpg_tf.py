import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

import gym
import numpy as np
import random
import tensorflow as tf

import rl
import rl.tf

from rl import \
    Framework, \
    run, \
    rollout, \
    LinearPolicyFactory, \
    InputFactory, \
    ProbabilityDistributionTypeFactory, \
    Episodes


class VanillaPolicyGradient:

    FRAMEWORK = Framework.TENSORFLOW

    def __init__(self, environment, random_seed, policy_factory, rollout_factory, min_steps_per_batch):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._rollout_factory = rollout_factory
        self._min_steps_per_batch = min_steps_per_batch
        self._observation_space = environment.observation_space
        self._action_space = environment.action_space

        self._learning_rate = 1e-2

        self._graph = None
        self._session = None
        self._policy = None

        self._graph = tf.Graph()
        with self._graph.as_default():

            tf.set_random_seed(self._random_seed)
            # set random seed here, or somewhere
            self._session = tf.Session(graph=self._graph)
            self._policy = policy_factory.create_policy(
                framework=self.FRAMEWORK,
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
            episode = self._rollout_factory(
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

environment_name = 'CartPole-v0'
# environment_name = 'MountainCar-v0'
# environment_name = 'Pendulum-v0'
random_seed = 0
max_epochs = 10000
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

def algorithm_function(environment):
    policy_factory = LinearPolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())
    return VanillaPolicyGradient(
        environment=environment,
        random_seed=random_seed,
        policy_factory=policy_factory,
        rollout_factory=rollout,
        min_steps_per_batch=1)

if __name__ == '__main__':
    run(
        algorithm_function=algorithm_function,
        environment_function=environment_function,
        specification=specification,
        random_seed=random_seed,
        max_epochs=max_epochs,
        deterministic=True)
