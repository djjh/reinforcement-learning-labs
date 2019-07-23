import numpy as np

from . import AdvantageFunction
from rl.core import discount_cumsum

class Gae(AdvantageFunction):

    def __init__(self, value_function, gamma, lambduh):
        self._value_function = value_function
        self._gamma = gamma
        self._lambda = lambduh

    def get_advantages(self, episodes):
        return self._get_batch_advantages(episodes)

    def update(self, episodes):
        with self._value_function:
            self._value_function.update(episodes)

    def _get_batch_advantages(self, episodes):
        return [advantage
            for episode in episodes
            for advantage in self._get_advantages(episode)]

    def _get_advantages(self, episode):
        with self._value_function:
            rewards = episode.get_rewards()
            values = self._value_function.get_values(episode)
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            advantages = discount_cumsum(rewards, self._gamma * self._lambda)
            return advantages

    def _get_batch_returns(self, episodes):
        return [weight
            for episode in episodes
            for weight in self._get_returns(episode)]

    def _get_returns(self, episode):
        rewards = episode.get_rewards()
        returns = discount_cumsum(rewards)
        return returns
