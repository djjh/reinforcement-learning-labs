import numpy as np

from scipy.special import softmax


class CategoricalProbabilityDistribution:

    def __init__(self, logits):
        self._logits = logits
        self._probabilities = softmax(logits)

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
