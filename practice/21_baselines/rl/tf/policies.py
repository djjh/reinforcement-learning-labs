import gym
import numpy as np
import random
import scipy.special

from rl import Framework


class LinearPolicy:

    FRAMEWORK = Framework.TENSORFLOW

    def __init__(self, observation_space, action_space, pd_factory_factory, session):
        self.observation_space = observation_space
        self.action_space = action_space
        self.pd_factory = pd_factory_factory.get_probability_distribution_factory(FRAMEWORK, action_space)
        self.session = session
        self.policy = tf.layers.Dense(units=self.action_dimensions, activation=None, use_bias=False)(self.observations)
        self.probability_distribution = self.pd_factory.create_probability_distribution(self.policy)


    def get_framework(self):
        return LinearPolicy.FRAMEWORK

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError();

    def action(self, observation, deterministic):
        # observation = observation.reshape(-1, 1)
        if deterministic:
            action = self.session.run(
                fetches=self.deterministic_action,
                feed_dict={self.observations: observation})
        else:
            action = self.session.run(
                fetches=self.action,
                feed_dict={self.observations: observation})
        return action

    def step(self, observation, deterministic):
        action = self.action(observation, deterministic)
        probabilities = self.session.run(
                fetches=self.policy_probability,
                feed_dict={self.observations: observation})
        return action, probabilities
