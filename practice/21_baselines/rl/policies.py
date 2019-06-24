import gym
import numpy as np
import random
import scipy.special

import rl
import rl.np
import rl.tf
from rl import Framework

class RecordingPolicy:

    def __init__(self, policy):
        self._policy = policy
        self._probabilities = []

    def action(self, observation, deterministic):
        action, probability_distribution = self._policy.step(observation, deterministic)
        self._probabilities.append(probability_distribution.probabilities())
        return action

    def get_probabilities(self):
        return self._probabilities


class LinearPolicyFactory:

    def __init__(self, input_factory, distribution_type_factory):
        self._input_factory = input_factory
        self._distribution_type_factory = distribution_type_factory

    def create_policy(self, framework, observation_space, action_space, session):
        if framework == Framework.SCRATCH:
            return rl.np.LinearPolicy(
                observation_space=observation_space,
                action_space=action_space,
                input_factory=self._input_factory,
                distribution_type_factory=self._distribution_type_factory)
        elif framework == Framework.TENSORFLOW:
            return rl.tf.LinearPolicy(
                observation_space=observation_space,
                action_space=action_space,
                input_factory=self._input_factory,
                distribution_type_factory=self._distribution_type_factory,
                session=session)
