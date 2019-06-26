import numpy as np
import tensorflow as tf

from rl.tf.distributions import CategoricalProbabilityDistribution

class CategoricalProbabilityDistributionType:

    def __init__(self, n_categories):
        self._n_categories = n_categories

        self._parameter_shape = (self._n_categories,)
        self._flat_parameter_length = np.prod(self._parameter_shape)
        self._sample_shape = ()
        self._dtype = tf.int32

    def create_probability_distribution(self, logits):
        self._check_logits_shape(logits)
        return CategoricalProbabilityDistribution(self, logits)

    def parameter_shape(self):
        return self._parameter_shape

    def get_flat_parameter_length(self):
        return self._flat_parameter_length

    def sample_shape(self):
        return self._sample_shape

    def sample_dtype(self):
        return self._dtype

    def _check_logits_shape(self, logits):
        logits_shape = logits.get_shape()[1:]
        assert logits_shape == self._parameter_shape, \
            "Invalid logits shape: {} vs expected {}.".format(
                logits_shape,
                self._parameter_shape)
