import nevergrad as ng
import numpy as np

from rl.core import Experience


class Nevergrad:

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
            observation_space=self._observation_space,
            action_space=self._action_space)

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
