import gym
import numpy as np
import random
import scipy.special

from rl import Framework


class CategoricalProbabilityDistributionFactory:

    FRAMEWORK = Framework.TENSORFLOW

    def __init__(self, n_categories):
        self.n_categories = n_categories

    def create_probability_distribution(self, logits):
        return CategoricalProbabilityDistribution(logits)

    def parameter_shape(self):
        return (self.n_categories,)

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...


class CategoricalProbabilityDistribution:

    FRAMEWORK = Framework.TENSORFLOW

    def __init__(self, logits):
        self.logits = logits

    def mode(self):
        # TODO: Move this comment/documentation somewhere else?
        # The most frequent value is the same as argmax, e.g.
        # for PMF [0.1, 0.6, 0.3]
        # index 0 will be sampled 0.1 of the time,
        # index 1 will be sampled 0.6 of the time,
        # and index 2 will be sampled 0.3 of the time.
        # Thus the mode is the index of the value with the highest value,
        # aka argmax.
        return tf.argmax(self.logits, axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)
