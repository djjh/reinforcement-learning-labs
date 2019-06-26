import numpy as np


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
