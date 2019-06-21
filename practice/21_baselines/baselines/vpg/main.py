import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

import gym
import nevergrad as ng
import numpy as np
import random

from rl import *


class RecordingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.probabilities = []
    def action(self, observation, deterministic):
        action, probability_distribution = self.policy.step(observation, deterministic)
        self.probabilities.append(probability_distribution.probabilities)
        return action

class VPG:

    def __init__(self, environment, random_seed, policy_factory, create_rollout, min_steps_per_batch):
        self.environment = environment
        self.random_seed = random_seed
        self.policy_factory = policy_factory
        self.create_rollout = create_rollout
        self.min_steps_per_batch = min_steps_per_batch
        self.deterministic_update_policy = False

        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.policy = self.policy_factory.create(
            observation_space=self.observation_space,
            action_space=self.action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        self.policy_return = -np.inf
        self.policy_steps = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        return self.policy.action(observation, deterministic)

    def update(self):
        episodes = Episodes()
        episodes_probabilities = []

        while episodes.num_steps() < self.min_steps_per_batch:
            recording_policy = RecordingPolicy(self.policy)
            episode = self.create_rollout(
                self.environment,
                recording_policy,
                random_seed=self.random_seed,
                deterministic=self.deterministic_update_policy,
                render=False)
            episodes.append(episode)
            episodes_probabilities.append(recording_policy.probabilities)

        grads = []
        rewards = []
        for i in range(len(episodes)):
            episode_grads = []
            episode_rewards = []
            episode = episodes[i]
            episode_probabilities = episodes_probabilities[i]
            for j in range(len(episode)):
                observation = episode.observations[j]
                action = episode.actions[j]
                reward = episode.rewards[j]
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
                self.policy.model += 0.0025 * grads[i][j] * sum([ r * (0.99 ** r) for t,r in enumerate(rewards[i][j:])])

        # print(self.policy.model)

environment_name = 'CartPole-v0'
# environment_name = 'MountainCar-v0'
# environment_name = 'Pendulum-v0'
random_seed = 0
max_epochs = 1000
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

def algorithm_function(environment):
    return VPG(
        environment=environment,
        random_seed=random_seed,
        policy_factory=LinearPolicyFactory(),
        create_rollout=rollout,
        min_steps_per_batch=200)

if __name__ == '__main__':
    run(algorithm_function, environment_function, specification, random_seed, max_epochs)
