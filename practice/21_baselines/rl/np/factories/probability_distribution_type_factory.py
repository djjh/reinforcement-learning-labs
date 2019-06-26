import gym
import numpy as np
import random
import scipy.special
import rl

from gym.spaces import Box
from gym.spaces import Discrete
from rl.np.distributions import CategoricalProbabilityDistributionType
from rl.np.distributions import DiagonalGaussianProbabilityDistributionType

class ProbabilityDistributionTypeFactory:

    def create_probability_distribution_type(self, space):
        if isinstance(space, Discrete):
            return CategoricalProbabilityDistributionType(space.n)
        elif isinstance(space, Box):
            return DiagonalGaussianProbabilityDistributionType(space.shape)
        else:
            raise NotImplementedError()
