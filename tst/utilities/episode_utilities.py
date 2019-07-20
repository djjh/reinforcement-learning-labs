import pytest
import rl

from rl.core import Episode
from rl.core import Episodes
from rl.tf.advantages import AdvantageFunction


####################
# Helper Functions #
####################

def generate_episode(rewards):
    observation = None
    action = None
    episode = Episode()
    for reward in rewards:
        episode.append(observation, action, reward)
    return episode

def generate_episodes(batch_rewards):
    episodes = Episodes()
    for rewards in batch_rewards:
        episodes.append(generate_episode(rewards))
    return episodes
