import gym
import numpy as np
import random
import scipy.special

from rl import *

class RecordingPolicy:
    def __init__(self, policy):
        self._policy = policy
        self._probabilities = []
    def action(self, observation, deterministic):
        action, probability_distribution = self._policy.step(observation, deterministic)
        self._probabilities.append(probability_distribution.probabilities)
        return action


class LinearPolicyFactory:

    def create_policy(self, framework, observation_space, action_space,
                input_factory, distribution_factory_factory, session):
        if framework == Framework.SCRATCH:
            return rl.np.LinearPolicy(
                observation_space=observation_space,
                action_space=action_space,
                input_factory=input_factory,
                distribution_factory_factory=distribution_factory_factory)
        elif framework == Framework.TENSORFLOW:
            return rl.tf.LinearPolicy(
                observation_space=observation_space,
                action_space=action_space,
                input_factory=input_factory,
                distribution_factory_factory=distribution_factory_factory,
                session=session)
