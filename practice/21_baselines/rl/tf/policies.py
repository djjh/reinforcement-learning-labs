import gym
import numpy as np
import random
import scipy.special
import tensorflow as tf

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete
from rl import Framework


class LinearPolicy:

    FRAMEWORK = Framework.TENSORFLOW

    def __init__(self, observation_space, action_space, input_factory, distribution_factory_factory, session):
        self._observation_space = observation_space
        self._action_space = action_space
        self._input_factory = input_factory
        self._distribution_factory_factory = distribution_factory_factory
        self._session = session

        self._distribution_factory = self._distribution_factory_factory.create_probability_distribution_factory(
            framework=self.FRAMEWORK,
            space=action_space)
        self._input = input_factory.create(
            framework=self.FRAMEWORK,
            space=self._observation_space,
            batch_size=None)

        self._model = self._create_model(self._input, self._distribution_factory)
        self._probability_distribution = self._distribution_factory.create_probability_distribution(self._model)

        self._observations =self._input.get_input()
        self._actions = self._probability_distribution.sample()
        self._deterministic_actions = self._probability_distribution.mode()
        self._log_probabilities = self._probability_distribution.log_probabilities(self._actions)

    def _create_model(self, input, distribution_factory):

        # Is this the right thing to do for non-categorical action spaces?
        units = distribution_factory.get_flat_parameter_length()

        previous_layer = input.get_input()
        return  tf.layers.Dense(units=units)(previous_layer)

    def get_framework(self):
        return LinearPolicy.FRAMEWORK

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError();

    def get_observations(self):
        return self._observations

    def get_observations(self):
        return self._observations

    def get_actions(self):
        return self._actions

    def get_deterministic_actions(self):
        return self._deterministic_actions


    def get_log_probabilities(self):
        return self._log_probabilities

    # Non-batch
    def action(self, observation, deterministic):
        observation = observation.reshape(1, -1)
        if deterministic:
            action = self._session.run(
                fetches=self._deterministic_actions,
                feed_dict={self._observations: observation})
        else:
            action = self._session.run(
                fetches=self._actions,
                feed_dict={self._observations: observation})
        return action[0]

    def step(self, observation, deterministic):
        action = self._action(observation, deterministic)
        probabilities = self._session.run(
                fetches=self._policy_probability,
                feed_dict={self._observations: observation})
        return action, probabilities
