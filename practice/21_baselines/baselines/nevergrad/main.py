import sys, os
from pathlib import Path

# For now we can operate this way...
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

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

class Nevergrad:

    FRAMEWORK = Framework.SCRATCH

    def __init__(self, environment, random_seed, policy_factory, create_rollout, optimizer, budget, low, high):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._create_rollout = create_rollout
        self._budget = budget
        self._low = low
        self._high = high
        self._deterministic_update_policy = True

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space
        self._policy = self._create_policy()
        self._policy_return = -np.inf
        self._policy_steps = -np.inf

        self._shape = self._policy.get_parameters()['model'].shape
        self._dims = np.prod(self._shape)
        instrumentation = ng.Instrumentation(ng.var.Array(self._dims).bounded(low, high))
        self._optimizer = optimizer(instrumentation=instrumentation, budget=budget)

    def _create_policy(self):
        return self._policy_factory.create_policy(
            framework=self.FRAMEWORK,
            observation_space=self._observation_space,
            action_space=self._action_space,
            session=None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        return self._policy.action(observation, deterministic)

    def update(self):

        def rewards(parameters):
            parameters = np.array(parameters).reshape(self._shape)
            # May want to include multiple iterations here in case of a stochastic environment or policy.
            policy = self._create_policy()
            policy.set_parameters({'model': parameters})
            episode = self._create_rollout(
                self._environment,
                policy,
                random_seed=self._random_seed,
                deterministic=self._deterministic_update_policy,
                render=False)
            return -episode.get_return()  # nevergrad optimizers minimize!

        recommendation = self._optimizer.optimize(rewards)
        # print(recommendation)
        parameters = recommendation.args[0]
        parameters = np.array(parameters).reshape(self._shape)

        policy = self._create_policy()
        policy.set_parameters({'model': parameters})
        self._policy = policy

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
    policy_factory = LinearPolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())
    return Nevergrad(
        environment=environment,
        random_seed=random_seed,
        policy_factory=policy_factory,
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
    run(
        algorithm_function=algorithm_function,
        environment_function=environment_function,
        specification=specification,
        random_seed=random_seed,
        max_epochs=max_epochs,
        deterministic=True)



if __name__ == '__main__':
    main()
