import gym
import numpy as np
import random
import scipy.special

import rl

from rl import Framework

class ProbabilityDistributionTypeFactory:

    def create_probability_distribution_type(self, framework, space):
        if framework == Framework.SCRATCH:
            if isinstance(space, gym.spaces.Discrete):
                return rl.np.CategoricalProbabilityDistributionType(space.n)
            elif isinstance(space, gym.spaces.Box):
                return rl.np.DiagonalGaussianProbabilityDistributionType(space.shape)
            else:
                raise NotImplementedError()
        elif framework == Framework.TENSORFLOW:
            if isinstance(space, gym.spaces.Discrete):
                return rl.tf.CategoricalProbabilityDistributionType(space.n)
            else:
                raise NotImplementedError()
