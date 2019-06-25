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

def rollout(environment, policy, random_seed, deterministic, render):
    episode = Episode()
    if deterministic:
    #     # TODO: should replace this with supplying the seed instead, and
    #     # should reseed all random seeds here.
    #     np.random.seed(random_seed)
    #     random.seed(random_seed)
        environment.seed(random_seed)
    observation = environment.reset()
    while True:
        if render:
            environment.render()
        action = policy.action(observation, deterministic)
        next_observation, reward, done, info = environment.step(action)
        episode.append(observation, action, reward)
        if done:
            break;
        else:
            observation = next_observation
    return episode
