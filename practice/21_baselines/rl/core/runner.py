import gym
import numpy as np
import random

from rl.core import Rollout

class Runner:

    def __init__(self, algorithm_factory, environment_factory):
        self._algorithm_factory = algorithm_factory
        self._environment_factory = environment_factory

    def run(self, algorithm_name, environment_name, random_seed, max_epochs, deterministic):

        np.random.seed(random_seed)
        random.seed(random_seed)

        specification, environment = self._environment_factory.create_environment(environment_name)
        environment.seed(random_seed)

        algorithm = self._algorithm_factory.create_algorithm(algorithm_name, environment, random_seed)

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
                    episode = Rollout(environment, policy, random_seed=random_seed, deterministic=deterministic, render=False)
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
            episode = Rollout(environment,  policy, random_seed=random_seed, deterministic=deterministic, render=True)
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
