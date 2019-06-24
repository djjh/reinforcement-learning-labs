import gym
import numpy as np
import random
import scipy.special

import rl


class CategoricalProbabilityDistributionType:

    def __init__(self, n_categories):
        self._n_categories = n_categories

    def probability_distribution(self, logits):
        return CategoricalProbabilityDistribution(logits)

    def parameter_shape(self):
        return (self._n_categories,)

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...


class CategoricalProbabilityDistribution:

    def __init__(self, logits):
        self._logits = logits
        self._probabilities = scipy.special.softmax(logits)

    def mode(self):
        # The most frequent value is the same as argmax, e.g.
        # for PMF [0.1, 0.6, 0.3]
        # index 0 will be sampled 0.1 of the time,
        # index 1 will be sampled 0.6 of the time,
        # and index 2 will be sampled 0.3 of the time.
        # Thus the mode is the index of the value with the highest value,
        # aka argmax.
        return np.argmax(self._logits)

    def sample(self):
        return np.random.choice(len(self._probabilities), p=self._probabilities)

    def probabilities(self):
        return self._probabilities


class DiagonalGaussianProbabilityDistributionType:

    def __init__(self, shape):
        self._shape = shape

    def probability_distribution(self, mean):
        return DiagonalGaussianProbabilityDistribution(mean)

    def parameter_shape(self):
        return self._shape

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...

class DiagonalGaussianProbabilityDistribution:

    def __init__(self, mean):
        self._mean = mean
        self._logstd = 0.0
        self._std = 0.0 #np.exp(self._logstd)

    def mode(self):
        return self._mean

    def sample(self):
        return self._mean.shape + self._std * np.random.standard_normal(self._mean.shape)

    def probabilities(self):
        raise NotImplementedError()
