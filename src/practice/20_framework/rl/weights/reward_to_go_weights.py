import logging
import numpy as np

from rl.weights import Weights

logger = logging.getLogger(__name__)

class RewardToGoWeights(Weights):

    def __init__(self, discount):
        self.discount = discount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_batch_weights(self, episodes):
        return [weight for episode in episodes for weight in self.get_weights(episode)]

    def get_lstm_batch_weights(self, episodes):
        weights = np.array([[weight for weight in self.get_weights(episode)] for episode in episodes])
        weights = np.swapaxes(weights, 0, 1)
        logger.debug("Weights shape: {}".format(weights.shape))
        return weights

    def get_weights(self, episode):
        rewards = episode.rewards
        output = [0] * len(rewards)
        for i in range(len(rewards)-1, 0, -1):
            output[i] += rewards[i]
            output[i-1] += (1.0 - self.discount) * output[i]
        return output

    def update(self, epoch, episodes):
        pass
