import tensorflow as tf

class CategoricalProbabilityDistribution:

    def __init__(self, factory, logits):
        self._factory = factory
        self._logits = logits

        self._n_categories = factory._n_categories

    def mode(self):
        # TODO: Move this comment/documentation somewhere else?
        # The most frequent value is the same as argmax, e.g.
        # for PMF [0.1, 0.6, 0.3]
        # index 0 will be sampled 0.1 of the time,
        # index 1 will be sampled 0.6 of the time,
        # and index 2 will be sampled 0.3 of the time.
        # Thus the mode is the index of the value with the highest value,
        # aka argmax.
        return tf.argmax(self._logits, axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self._logits), dtype=self._logits.dtype)
        return tf.argmax(self._logits - tf.log(-tf.log(uniform)), axis=-1)


    def log_probabilities(self, actions):
        action_masks = tf.one_hot(indices=actions, depth=self._n_categories)
        return tf.reduce_sum(action_masks * tf.nn.log_softmax(self._logits), axis=1)
