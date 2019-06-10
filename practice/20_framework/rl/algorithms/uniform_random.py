import numpy as np

from rl.core import Algorithm, Policy, PolicyFactory


class UniformRandom(Algorithm):

    def __init__(
            self,
            environment,
            policy_factory: PolicyFactory,
            rollout_function,
            batch_size,
            low,
            high):
        self.environment = environment
        self.policy_factory = policy_factory
        self.rollout_function = rollout_function
        self.batch_size = batch_size
        self.low = low
        self.high = high

        self.policy = policy_factory.create_policy(self.environment)
        self.policy_return = -np.inf

    def get_policy(self) -> Policy:
        return self.policy

    def update(self):

        def uniform_sample_new_parameters(ps):
            return [np.random.uniform(self.low, self.high, p.shape) for p in ps]

        best_return = -np.inf
        best_policy = None
        parameters = self.policy.get_parameters()

        for i in range(self.batch_size):

            policy = self.policy_factory.create_policy(self.environment)
            policy.set_parameters(uniform_sample_new_parameters(parameters))
            episode = self.rollout_function(self.environment, policy, render=False)

            episode_return = episode.get_return()
            if episode_return > best_return:
                best_return = episode_return
                best_policy = policy
        if best_return > self.policy_return:
            self.policy_return = best_return
            self.policy = best_policy
