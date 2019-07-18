import traceback
import rl

import gym
import numpy as np
import random

from rl.core.graph import *

from rl.core import Rollout

from rl.core import AlgorithmFactory
from rl.core import EnvironmentFactory
from rl.core import Runner


def main():
    errors = []
    for graph in graphs():
        algorithm_class_name = fullname(graph.get('algorithm'))
        environment_name = graph.get('environment_name')
        print('---- {} - {} ----\n'.format(algorithm_class_name, environment_name))
        try:
            graph.get('experiment').run()
        except (KeyboardInterrupt, SystemExit):
            print()
            exit()
        except:
            errors.append({'algorithm': algorithm_class_name, 'exception': traceback.format_exc()})

    if len(errors) > 0:
        print('Failed:')

    for problem in errors:
        print('START------------------------------------')
        print(problem['algorithm'])
        print(problem['exception'])
        print('END------------------------------------')

def fullname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


class InMemoryLogger():

    def __init__(self):
        self._info_logs = []
        self._metrics = {}

    def info(self, message):
        self._info_logs.append(message)

    def metric(self, metric, value):
        if not metric in self._metrics:
            self._metrics[metric] = []
        self._metrics[metric].append(value)

    def plot(self, mx, my):
        pass  # plot(self._metrics[mx], self._metrics[my])

# class FileAppendingLogger():
#     pass

class Experiment():

    def __init__(self, algorithm, specification, environment, log, random_seed, max_epochs, deterministic):
        self.algorithm = algorithm
        self.specification = specification
        self.environment = environment
        self.log = log
        self.random_seed = random_seed
        self.max_epochs = max_epochs
        self.deterministic = deterministic

    def run(self):
        algorithm = self.algorithm
        specification = self.specification
        environment = self.environment
        random_seed = self.random_seed
        max_epochs = self.max_epochs
        deterministic = self.deterministic

        if deterministic:
            np.random.seed(random_seed)
            random.seed(random_seed)
            environment.seed(random_seed)

        with algorithm, environment:

            max_episode_steps = specification.max_episode_steps
            reward_threshold = specification.reward_threshold
            has_reward_threshold = reward_threshold is not None

            for epoch in range(1, max_epochs+1):

                self.log.metric("epoch", epoch)

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

                self.log.metric("win_count", win_count)
                self.log.metric("win", win)
                self.log.metric("ep2", episode_reward)

                print('                                                                                 ',
                    end="\r")
                print('epoch: {}, wins: {}, length: {}, reward: {}'.format(epoch, win_count, np.mean(episode_steps), np.mean(episode_rewards)),
                    end="\r")

                if win:
                    break

            self.log.plot('epoch', 'ep2')

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


def graphs():
    return GraphGenerator(specifications=[
        Specification(providers=[
            ValueNode(
                name='random_seed',
                value=0
            ),
            ValuesNode(
                name='environment_name',
                values=[
                    # 'CartPole-v0',
                    # 'CartPole-v1',
                    'MountainCar-v0'
                    # 'Pendulum-v0'  # Not NotImplementedError
                ]
            ),
            FunctionNode(
                name='in-memory-logger',
                function=InMemoryLogger
            ),
            FunctionNode(
                name='environment',
                function=gym.make,
                kwargs={'id': InjectNode(name='environment_name')}
            ),
            # Here is one possible way of unwrapping properties of dependencies:
            # FunctionNode(
            #     name='observation_space',
            #     function=lambda e: e.observation_space,
            #     kwargs={'e': InjectNode(name='environment')}),
            FunctionNode(
                name='specification',
                function=gym.spec,
                kwargs={'id': InjectNode(name='environment_name')}
            ),
            FunctionNode(
                name='tensorflow_input_factory',
                function=rl.tf.factories.InputFactory
            ),
            FunctionNode(
                name='tensorflow_policy_factory',
                function=rl.tf.factories.PolicyFactory,
                kwargs={
                    'input_factory': InjectNode(name='tensorflow_input_factory'),
                    'distribution_type_factory': FunctionNode(function=rl.tf.factories.ProbabilityDistributionTypeFactory),
                    'learning_rate': ValuesNode(values=[5e-3])
                }
            ),
            FunctionNode(
                name='numpy_policy_factory',
                function=rl.np.factories.PolicyFactory,
                kwargs={
                    'input_factory': FunctionNode(function=rl.np.factories.InputFactory),
                    'distribution_type_factory': FunctionNode(function=rl.np.factories.ProbabilityDistributionTypeFactory),
                }
            ),
            NodesNode(
                name='tensorflow_advantage_function',
                nodes=[
                    FunctionNode(
                        function=rl.tf.advantages.GeneralizedAdvantageEstimationFunction,
                        kwargs={
                            'value_function': FunctionNode(
                                function=rl.tf.values.LinearValueFunction,
                                kwargs={
                                    'environment': InjectNode(name='environment'),
                                    'input_factory': InjectNode(name='tensorflow_input_factory'),
                                    'iterations': ValueNode(value=10),
                                    'learning_rate': ValueNode(value=1e-2)
                                }
                            )
                        }
                    ),
                    # FunctionNode(function=rl.tf.advantages.CumulativeRewardAdvantageFunction),
                    # FunctionNode(
                    #     function=rl.tf.advantages.RewardToGoAdvantageFunction,
                    #     kwargs={
                    #         'discount': ValuesNode(values=[0.5])
                    #     }
                    # ),
                ]
            ),
            NodesNode(
                name='algorithm',
                nodes=[
                    # In Progres ....
                    # FunctionNode(
                    #     name='tensorflow_vanilla_policy_gradient',
                    #     function=rl.tf.algorithms.VanillaPolicyGradient,
                    #     kwargs=[
                    #         {
                    #             'environment': InjectNode(name='environment'),
                    #             'random_seed': InjectNode(name='random_seed'),
                    #             'policy_factory': InjectNode(name='tensorflow_policy_factory'),
                    #             'advantage_function': InjectNode(name='tensorflow_advantage_function'),
                    #             'Rollout': ValueNode(value=rl.core.Rollout),
                    #             'min_steps_per_batch': ValuesNode(values=[
                    #                 1,
                    #                 # 100,
                    #             ])
                    #         }
                    #     ]
                    # ),
                    FunctionNode(
                        name='numpy_vanilla_policy_gradient',
                        function=rl.np.algorithms.VanillaPolicyGradient,
                        kwargs=[
                            {
                                'environment': InjectNode(name='environment'),
                                'random_seed': InjectNode(name='random_seed'),
                                'policy_factory': InjectNode(name='numpy_policy_factory'),
                                'rollout_factory': ValueNode(value=rl.core.Rollout),
                                'min_steps_per_batch': ValuesNode(values=[1, 100])
                            }
                        ]
                    ),
                    # FunctionNode(
                    #     name='numpy_random_search',
                    #     function=rl.np.algorithms.RandomSearch,
                    #     kwargs=[
                    #         {
                    #             'environment': InjectNode(name='environment'),
                    #             'random_seed': InjectNode(name='random_seed'),
                    #             'policy_factory': InjectNode(name='numpy_policy_factory'),
                    #             'create_rollout': ValueNode(value=rl.core.Rollout),
                    #             'batch_size': ValuesNode(values=[1, 100]),
                    #             'explore': ValuesNode(values=[0.25, 0.5, 0.75])
                    #         }
                    #     ]
                    # )
                ]
            ),
            FunctionNode(
                name='experiment',
                function=Experiment,
                kwargs={
                    'specification': InjectNode(name='specification'),
                    'environment': InjectNode(name='environment'),
                    'algorithm': InjectNode(name='algorithm'),
                    'log': InjectNode(name='in-memory-logger'),
                    'random_seed': InjectNode(name='random_seed'),
                    'max_epochs': ValueNode(value=200),
                    'deterministic': ValueNode(value='True')
                })
        ]),
    ])




if __name__ == '__main__':
    main()
