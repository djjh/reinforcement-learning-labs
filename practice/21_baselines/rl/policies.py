import gym
import numpy as np
import random
import rl
import scipy.special


class RecordingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.probabilities = []
    def action(self, observation, deterministic):
        action, probability_distribution = self.policy.step(observation, deterministic)
        self.probabilities.append(probability_distribution.probabilities)
        return action


class LinearPolicyFactory:

    def create(self, framework, observation_space, action_space, pd_factory_factory):
        if framework == Framework.SCRATCH:
            return rl.np.LinearPolicy(observation_space, action_space, pd_factory_factory)
        elif framework == Framework.TENSORFLOW:
            return rl.tf.LinearPolicy(observation_space, action_space, pd_factory_factory)
