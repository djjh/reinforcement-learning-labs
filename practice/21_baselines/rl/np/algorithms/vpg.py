import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..', '..')).resolve()))

import gym
import nevergrad as ng
import numpy as np
import random
import rl

from rl import Episodes
from rl import Framework
from rl import InputFactory
from rl import LinearPolicyFactory
from rl import ProbabilityDistributionTypeFactory
from rl import RecordingPolicy
from rl import rollout
from rl import run


class VanillaPolicyGradient:

    FRAMEWORK = Framework.SCRATCH

    def __init__(self, environment, random_seed, policy_factory, create_rollout, min_steps_per_batch):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._create_rollout = create_rollout
        self._min_steps_per_batch = min_steps_per_batch
        self._deterministic_update_policy = False

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space
        self._policy = self._policy_factory.create_policy(
            framework=self.FRAMEWORK,
            observation_space=self._observation_space,
            action_space=self._action_space,
            session=None)  # Better way to handle this?
        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        return self._policy.action(observation, deterministic)

    def update(self):
        episodes = Episodes()
        episodes_probabilities = []

        while episodes.num_steps() < self._min_steps_per_batch:
            recording_policy = RecordingPolicy(self._policy)
            episode = self._create_rollout(
                self._environment,
                recording_policy,
                random_seed=self._random_seed,
                deterministic=self._deterministic_update_policy,
                render=False)
            episodes.append(episode)
            episodes_probabilities.append(recording_policy.get_probabilities())

        grads = []
        rewards = []
        for i in range(len(episodes)):
            episode_grads = []
            episode_rewards = []
            episode = episodes[i]
            episode_probabilities = episodes_probabilities[i]
            for j in range(len(episode)):
                observation = episode.get_observations()[j]
                action = episode.get_actions()[j]
                reward = episode.get_rewards()[j]
                probabilities = episode_probabilities[j]

                softmax = probabilities
                s = softmax.reshape(-1,1)
                grad_softmax = (np.diagflat(s) - np.dot(s, s.T))
                grad_softmax = grad_softmax[action,:]
                grad_log = grad_softmax / softmax[action]

                episode_grads.append(grad_log[None,:].T.dot(observation[None,:]))
                episode_rewards.append(reward)
            grads.append(episode_grads)
            rewards.append(episode_rewards)

        for i in range(len(grads)):
            for j in range(len(grads[i])):
                self._policy._model += 0.0025 * grads[i][j] * sum([ r * (0.99 ** r) for t,r in enumerate(rewards[i][j:])])

        # print(self._policy.model)

environment_name = 'CartPole-v0'
# environment_name = 'MountainCar-v0'
# environment_name = 'Pendulum-v0'
random_seed = 0
max_epochs = 1000
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

def algorithm_function(environment):
    policy_factory = LinearPolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())
    return VPG(
        environment=environment,
        random_seed=random_seed,
        policy_factory=policy_factory,
        create_rollout=rollout,
        min_steps_per_batch=200)

if __name__ == '__main__':
    run(
        algorithm_function=algorithm_function,
        environment_function=environment_function,
        specification=specification,
        random_seed=random_seed,
        max_epochs=max_epochs,
        deterministic=True)
