import numpy as np

from rl.np.distributions import DiagonalGaussianProbabilityDistribution


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
