from . import AdvantageFunction

class CumulativeRewardAdvantageFunction(AdvantageFunction):

    def __init__(self):
        pass

    def get_advantages(self, experience):
        return self._get_batch_returns(experience)


    def update(self, experience):
        pass

    def _get_batch_returns(self, experience):
        return [episode_return
            for episode in experience
            for episode_return in self._get_returns(episode)]

    def _get_returns(self, episode):
        rewards = episode.get_rewards()
        return [sum(rewards)] * len(rewards)
