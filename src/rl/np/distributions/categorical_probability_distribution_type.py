import numpy as np

from rl.np.distributions import CategoricalProbabilityDistribution


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
        return np.int32
