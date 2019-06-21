import gym
import numpy as np
import random
import scipy.special

import rl

class ProbabilityDistributionFactoryFactory:

    def create_probability_distribution_factory(self, framework, space):
        if framework == SCRATCH:
            if isinstance(space, gym.spaces.Discrete):
                return rl.np.CategoricalProbabilityDistributionFactory(space.n)
            elif isinstance(space, gym.spaces.Box):
                return rl.np.DiagonalGaussianProbabilityDistributionFactory(space.shape)
            else:
                raise NotImplementedError()
        elif framework == TENSORFLOW:
            if isinstance(space, gym.spaces.Discrete):
                return rl.tf.CategoricalProbabilityDistributionFactory(space.n)
            else:
                raise NotImplementedError()
