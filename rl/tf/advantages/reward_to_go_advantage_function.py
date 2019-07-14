import numpy as np

from rl.core import discount_cumsum

class RewardToGoAdvantageFunction:

    def __init__(self, discount):
        self._discount = discount

    def get_advantages(self, episodes):
        return self._get_batch_returns(episodes)


    def update(self, experience):
        pass

    def _get_batch_returns(self, episodes):
        return [episode_return
            for episode in episodes
            for episode_return in self._get_returns(episode)]

    def _get_returns(self, episode):
        rewards = episode.get_rewards()
        return discount_cumsum(rewards, self._discount)
