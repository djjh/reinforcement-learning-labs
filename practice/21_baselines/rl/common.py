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


def run(algorithm_function, environment_function, specification, random_seed, max_epochs, deterministic):

    np.random.seed(random_seed)
    random.seed(random_seed)

    environment = environment_function()
    environment.seed(random_seed)

    algorithm = algorithm_function(environment)

    with algorithm, environment:

        max_episode_steps = specification.max_episode_steps
        reward_threshold = specification.reward_threshold
        has_reward_threshold = reward_threshold is not None

        for epoch in range(1, max_epochs+1):

            algorithm.update()

            episode_stepss = []
            episode_rewards = []
            required_wins = 100
            win_count = 0
            win = True

            while win and win_count < required_wins:
                policy = algorithm  # or should we have def policy(self) -> Policy ?
                episode = rollout(environment, policy, random_seed=random_seed, deterministic=deterministic, render=False)
                episode_steps = len(episode)
                episode_reward = episode.get_return()
                episode_stepss.append(episode_steps)
                episode_rewards.append(episode_reward)
                win = has_reward_threshold and episode_reward >= reward_threshold
                if win:
                    win_count += 1

            print('                                                                                 ',
                end="\r")
            print('epoch: {}, wins: {}, length: {}, reward: {}'.format(epoch, win_count, np.mean(episode_steps), np.mean(episode_rewards)),
                end="\r")

            if win:
                break

        policy = algorithm   # or should we have def policy(self) -> Policy ?
        episode = rollout(environment,  policy, random_seed=random_seed, deterministic=deterministic, render=True)
        episode_steps = len(episode)
        episode_reward = episode.get_return()

        print('Epochs: {}'.format(epoch))
        if has_reward_threshold:
            print('Target -> length: {}, return: {}'.format(max_episode_steps, reward_threshold))
            print('Actual -> length: {}, return: {}'.format(episode_steps, episode_reward))
            win = has_reward_threshold and episode_reward >= reward_threshold
            print('Win!' if win else 'Lose!')
        else:
            print('Max return: {}'.format(episode_reward))
        if specification.nondeterministic:
            print('The environment was nondeterministic, so we should check the mean.');

        if environment.viewer and environment.viewer.window:
            environment.viewer.window.set_visible(False)
            environment.viewer.window.dispatch_events()
