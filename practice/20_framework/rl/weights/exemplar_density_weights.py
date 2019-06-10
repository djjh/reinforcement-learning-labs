import numpy as np

from rl.weights import Weights
from rl.density import ExemplarDensity

def negative_log(x):
    return -np.log(x)

bonus_function = negative_log

class ExemplarDensityWeights(Weights):

    def __init__(self, environment, use_actions, log_directory, random_seed, learning_rate):
        self.environment = environment
        self.use_actions = use_actions
        self.log_directory = log_directory
        self.random_seed = random_seed
        self.learning_rate = learning_rate

        self.epoch_hack = 0
        self.bonus_coefficient = 0.1

        dimensions = np.prod(self.environment.observation_space.shape)
        dimensions += 1 if self.use_actions else 0

        self.density = ExemplarDensity(
            dimensions=dimensions,
            log_directory=self.log_directory,
            random_seed=self.random_seed,
            learning_rate=self.learning_rate)

    def __enter__(self):
        self.density.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.density.__exit__(exc_type, exc_val, exc_tb)
        pass

    def get_batch_weights(self, episodes):
        return [weight for weight in self.get_weights(episode) for episode in episodes]

    def get_lstm_batch_weights(self, episodes):
        weights = np.array([[weight for weight in self.get_weights(episode)] for episode in episodes])
        weights = np.swapaxes(weights, 0, 1)
        return weights

    def get_weights(self, episode):
        episode_rewards = episode.rewards
        x = self.get_input(episode=episode)
        density = self.density.estimate(epoch=self.epoch_hack, x=x)
        self.epoch_hack += 1
        bonuses = self.bonus_coefficient * bonus_function(density)
        episode_rewards = (np.array(episode.rewards) + bonuses).tolist()
        return np.cumsum(episode_rewards[::-1])[::-1].tolist()

    def get_input(self, episode):
        if self.use_actions:
            o = self.get_batch_observations(episode)
            a = np.expand_dims(self.get_batch_actions(episode), 1)
            oa = np.concatenate((o, a), axis=1)
            return oa.tolist()
        else:
            return self.get_batch_observations(episode)

    def get_batch_observations(self, episode):
        return np.array([observation for observation in episode.observations])

    def get_batch_actions(self, episode):
        return np.array([action for action in episode.actions])

    def update(self, epoch, episodes):
        self.density.update(epoch, episodes)
