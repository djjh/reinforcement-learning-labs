import gym
import numpy as np
import random

class Episode:
    def __init__(self):
        self._observations = []
        self._actions = []
        self._rewards = []

    def append(self, observation, action, reward):
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)

    def get_return(self):
        return sum(self._rewards)

    def __len__(self):
        return len(self._observations)

    def get_observations(self):
        return self._observations

    def get_actions(self):
        return self._actions

    def get_rewards(self):
        return self._rewards
