from . import AdvantageFunction

class Cumulative(AdvantageFunction):

    def __init__(self):
        pass

    def get_advantages(self, episodes):
        return self._get_batch_returns(episodes)


    def update(self, episodes):
        pass

    def _get_batch_returns(self, episodes):
        return [episode_return
            for episode in episodes
            for episode_return in self._get_returns(episode)]

    def _get_returns(self, episode):
        rewards = episode.get_rewards()
        return [sum(rewards)] * len(rewards)
