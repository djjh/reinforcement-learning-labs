import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))


import gym
import numpy as np
import random

from rl import *

class RandomSearch:

    def __init__(self, environment, random_seed, policy_factory, create_rollout, batch_size, explore):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._create_rollout = create_rollout
        self._batch_size = batch_size
        self._explore = explore
        self._deterministic_update_policy = True

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space
        self._policy = self._policy_factory.create(
            observation_space=self._observation_space,
            action_space=self._action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        action = self._policy.action(observation, deterministic)
        return action

    def update(self):
        def random_parameter(explore, p):
            return p + explore * np.random.standard_normal(p.shape)
        best_steps = -np.inf
        best_return = -np.inf
        best_policy = None
        parameters = self._policy.get_parameters()['model']
        for i in range(self._batch_size):
            policy = self._policy_factory.create(
                observation_space=self._observation_space,
                action_space=self._action_space,
                pd_factory_factory=ProbabilityDistributionFactoryFactory())
            policy.set_parameters({'model': random_parameter(self._explore, parameters)})
            episode = self._create_rollout(
                self._environment,
                policy,
                random_seed=self._random_seed,
                deterministic=self._deterministic_update_policy,
                render=False)
            episode_return = episode.get_return()
            episode_steps = len(episode)
            # print('\n', episode_return)
            if episode_return > best_return or episode_steps > best_steps:
                best_return = episode_return
                best_steps = episode_steps
                best_policy = policy
        if best_return >= self._policy_return or best_steps >= self._policy_steps:
            self._policy_return = best_return
            self._policy_steps = best_steps
            self._policy = best_policy

# def main():

environment_name = 'CartPole-v0'
# environment_name = 'MountainCar-v0'
# environment_name = 'Pendulum-v0'
random_seed = 0
max_epochs = 10000
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

# Wins at 24 epochs.
def algorithm_function(environment):
    return RandomSearch(
        environment=environment,
        random_seed=random_seed,
        policy_factory=LinearPolicyFactory(),
        create_rollout=rollout,
        batch_size=1,
        explore=0.5)


def main():
    run(algorithm_function, environment_function, specification, random_seed, max_epochs)



if __name__ == '__main__':
    main()
