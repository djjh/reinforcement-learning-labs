import sys, os
from pathlib import Path
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..')).resolve()))

import traceback
import rl

import gym
import numpy as np
import random

from rl.core import generate_functions

from rl.core import Rollout

from rl.core import AlgorithmFactory
from rl.core import EnvironmentFactory
from rl.core import Runner


def experiment(algorithm, specification, environment, random_seed, max_epochs, deterministic):
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


def generate_experiments(environment_name, random_seed):

    environment_factory = EnvironmentFactory()

    np.random.seed(random_seed)
    random.seed(random_seed)

    specification, environment = environment_factory.create_environment(environment_name)
    environment.seed(random_seed)

    args = {
        experiment: {
            'algorithm': [
                {
                    rl.tf.algorithms.VanillaPolicyGradient: {
                        'environment': [environment],
                        'random_seed': [random_seed],
                        'policy_factory': [
                            {
                                rl.tf.factories.PolicyFactory: {
                                    'input_factory': [{ rl.tf.factories.InputFactory: {} }],
                                    'distribution_type_factory': [{ rl.tf.factories.ProbabilityDistributionTypeFactory: {} }]
                                }
                            }
                        ],
                        'Rollout': [rl.core.Rollout],
                        'min_steps_per_batch': [1, 100, 1000],
                        'learning_rate': [1e-2, 5e-3, 1e-3]
                    }
                }
            ],
            'specification': [specification],
            'random_seed': [random_seed],
            'environment': [environment],
            'max_epochs': [200],
            'deterministic': [True]
        }
    }
    return generate_functions(args)


if __name__ == '__main__':

    problems = []
    environment_name = 'CartPole-v0'
    for experiment in generate_experiments(environment_name, 0):
        print('---- {} - {} ----\n'.format('something', environment_name))
        try:
            experiment()
        except:
            problems.append({'algorithm': 'somthing', 'exception': traceback.format_exc()})

    if len(problems) > 0:
        print('Failed:')
    for problem in problems:
        print('START------------------------------------')
        print(problem['algorithm'])
        print(problem['exception'])
        print('END------------------------------------')
