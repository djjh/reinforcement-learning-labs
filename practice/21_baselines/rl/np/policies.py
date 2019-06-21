import gym
import numpy as np
import random
import rl
import scipy.special

class LinearPolicy:

    FRAMEWORK = Framework.SCRATCH

    def __init__(self, observation_space, action_space, pd_factory_factory):
        self.observation_space = observation_space
        self.action_space = action_space

        self.pd_factory = pd_factory_factory.probability_distribution_factory(FRAMEWORK, action_space)

        self.observation_dimensions = np.prod(observation_space.shape)
        self.action_dimensions = np.prod(self.pd_factory.parameter_shape())

        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)

    def get_framework(self):
        return LinearPolicy.FRAMEWORK

    def get_parameters(self):
        return { 'model': np.array(self.model) }

    def set_parameters(self, parameters):
        self.model = np.array(parameters['model'])

    # Should we have setters for input, a step method, and getters for output?
    # e.g.
    #     + def observe(self, observation)
    #     + def step(self)
    #     + def probabilities(self) -> ProbabilityDistribution
    #     + def action(self) -> Action
    # or simply
    #     + def step(self, observation) -> Action, ProbabilityDistribution
    #
    # Should deterministic be an argument or should we have mode and sample methods?
    # e.g.
    #     + def mode(self, observation) -> Action
    #     + def sample(self, observation) -> Action
    # or simply
    #     + def action(self, observation, deterministic) -> Action
    #
    def action(self, observation, deterministic):
        observation = observation.reshape(-1, 1)
        action_logits = self.model.dot(observation).flatten()
        action_distribution = self.pd_factory.probability_distribution(action_logits)
        # print(self.model.shape, observation.shape, action_logits.shape)
        if deterministic:
            return action_distribution.mode()
        else:
            return action_distribution.sample()

    def step(self, observation, deterministic):
        observation = observation.reshape(-1, 1)
        action_logits = self.model.dot(observation).flatten()
        action_distribution = self.pd_factory.probability_distribution(action_logits)
        # print(self.model.shape, observation.shape, action_logits.shape)
        if deterministic:
            return action_distribution.mode(), action_distribution
        else:
            return action_distribution.sample(), action_distribution
