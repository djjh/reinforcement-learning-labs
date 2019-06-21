import gym
import numpy as np
import random
import scipy.special

import rl

class ProbabilityDistributionFactoryFactory:

    def probability_distribution_factory(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return CategoricalProbabilityDistributionFactory(space.n)
        elif isinstance(space, gym.spaces.Box):
            return DiagonalGaussianProbabilityDistributionFactory(space.shape)
        else:
            raise NotImplementedError()


class CategoricalProbabilityDistributionFactory:

    def __init__(self, n_categories):
        self.n_categories = n_categories

    def probability_distribution(self, logits):
        return CategoricalProbabilityDistribution(logits)

    def parameter_shape(self):
        return (self.n_categories,)

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...


class CategoricalProbabilityDistribution:

    def __init__(self, logits):
        self.logits = logits
        self.probabilities = scipy.special.softmax(logits)

    def mode(self):
        # The most frequent value is the same as argmax, e.g.
        # for PMF [0.1, 0.6, 0.3]
        # index 0 will be sampled 0.1 of the time,
        # index 1 will be sampled 0.6 of the time,
        # and index 2 will be sampled 0.3 of the time.
        # Thus the mode is the index of the value with the highest value,
        # aka argmax.
        return np.argmax(self.logits)

    def sample(self):
        return np.random.choice(len(self.probabilities), p=self.probabilities)


class DiagonalGaussianProbabilityDistributionFactory:

    def __init__(self, shape):
        self.shape = shape

    def probability_distribution(self, mean):
        return DiagonalGaussianProbabilityDistribution(mean)

    def parameter_shape(self):
        return self.shape

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...

class DiagonalGaussianProbabilityDistribution:

    def __init__(self, mean):
        self.mean = mean
        self.logstd = 0.0
        self.std = 0.0 #np.exp(self.logstd)

    def mode(self):
        return self.mean

    def sample(self):
        return self.mean.shape + self.std * np.random.standard_normal(self.mean.shape)
