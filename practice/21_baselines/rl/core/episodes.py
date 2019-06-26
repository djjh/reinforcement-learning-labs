import gym
import numpy as np
import random

class Episodes:
    def __init__(self):
        self._episodes = []
        self._cumulative_length = 0

    def __iter__(self):
        return iter(self._episodes)

    def __getitem__(self, index):
        return self._episodes[index]

    def __len__(self):
        return len(self._episodes)

    def append(self, episode):
        self._episodes.append(episode)
        self._cumulative_length += len(episode)

    def num_steps(self):
        return self._cumulative_length

    def get_batch_observations(self):
        return [observation
            for episode in self._episodes
            for observation in episode.observations]

    def get_batch_actions(self):
        return [action
            for episode in self._episodes
            for action in episode.actions]
