import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

import gym
import nevergrad as ng
import numpy as np
import random

from rl import *

class Nevergrad:

    def __init__(self, environment, random_seed, policy_factory, create_rollout, optimizer, budget, low, high):
        self.environment = environment
        self.random_seed = random_seed
        self.policy_factory = policy_factory
        self.create_rollout = create_rollout
        self.budget = budget
        self.low = low
        self.high = high
        self.deterministic_update_policy = True

        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.policy = self.policy_factory.create(
            observation_space=self.observation_space,
            action_space=self.action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        self.policy_return = -np.inf
        self.policy_steps = -np.inf

        self.shape = self.policy.get_parameters()['model'].shape
        self.dims = np.prod(self.shape)
        instrumentation = ng.Instrumentation(ng.var.Array(self.dims).bounded(low, high))
        self.optimizer = optimizer(instrumentation=instrumentation, budget=budget)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        return self.policy.action(observation, deterministic)

    def update(self):

        def rewards(parameters):

            parameters = np.array(parameters).reshape(self.shape)
            # May want to include multiple iterations here in case of a stochastic environment or policy.
            policy = self.policy_factory.create(
                observation_space=self.observation_space,
                action_space=self.action_space,
                pd_factory_factory=ProbabilityDistributionFactoryFactory())
            policy.set_parameters({'model': parameters})
            episode = self.create_rollout(
                self.environment,
                policy,
                random_seed=self.random_seed,
                deterministic=self.deterministic_update_policy,
                render=False)
            return -episode.get_return()  # nevergrad optimizers minimize!

        recommendation = self.optimizer.optimize(rewards)
        # print(recommendation)
        parameters = recommendation.args[0]
        parameters = np.array(parameters).reshape(self.shape)

        policy = self.policy_factory.create(
            observation_space=self.observation_space,
            action_space=self.action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        policy.set_parameters({'model': parameters})
        self.policy = policy

# environment_name = 'CartPole-v0'
environment_name = 'MountainCar-v0' # OnePlusOne gets there at least.
# environment_name = 'Pendulum-v0'
optimizer = ng.optimizers.OnePlusOne
# optimizer = ng.optimizers.TwoPointsDE
# optimizer = ng.optimizers.CMA
random_seed = 0
max_epochs = 1
specification = gym.spec(environment_name)

def environment_function():
    return gym.make(environment_name)

def algorithm_function(environment):
    return Nevergrad(
        environment=environment,
        random_seed=random_seed,
        policy_factory=LinearPolicyFactory(),
        create_rollout=rollout,
        optimizer=optimizer,
        budget=400,
        low=-1.0,
        high=1.0)

# print(specification.max_episode_seconds)
# print(specification.max_episode_steps)
# print(specification.nondeterministic)
# print(specification.reward_threshold)
# print(specification.tags)
# print(specification.timestep_limit)
# print(specification.trials)


def main():
    run(algorithm_function, environment_function, specification, random_seed, max_epochs)



if __name__ == '__main__':
    main()
