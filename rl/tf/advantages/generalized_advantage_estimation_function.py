import numpy as np

from rl.core import discount_cumsum

class GeneralizedAdvantageEstimationFunction:

    # TODO: (observation) or (observation, action) as input to the value function...
    def __init__(self, value_function):
        self._value_function = value_function

    def get_advantages(self, experience):
        print(type(experience))
        return self._get_batch_advantages(experience)

    def update(self, experience):
        self.value_function.update(experience)

    def _get_batch_advantages(self, experience):
        return [advantage
            for episode in experience
            for advantage in self._get_advantages(episode)]

    def _get_advantages(self, episode):
        rewards = episode.get_rewards()
        print(type(episode))
        values = self._value_function.get_values(episode)
        deltas = rewards[:-1] + 0.99 * values[1:] - values[:-1]
        print(deltas)
        return deltas

    def _get_batch_returns(self, experience):
        return [weight
            for episode in experience
            for weight in self._get_returns(episode)]

    def _get_returns(self, episode):
        rewards = episode.get_rewards()
        returns = discount_cumsum(rewards)
        print(returns.shape)
        return returns
