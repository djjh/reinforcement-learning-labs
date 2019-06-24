import gym
import numpy as np
import random
import rl
import scipy.special

class LinearPolicy:

    FRAMEWORK = Framework.SCRATCH

    def __init__(self, observation_space, action_space, pd_factory_factory):
        self._observation_space = observation_space
        self._action_space = action_space

        self._pd_factory = pd_factory_factory.probability_distribution_factory(FRAMEWORK, action_space)

        self._observation_dimensions = np.prod(observation_space.shape)
        self._action_dimensions = np.prod(self._pd_factory.parameter_shape())

        self._model = np.random.randn(self._action_dimensions, self._observation_dimensions)

    def get_framework(self):
        return LinearPolicy.FRAMEWORK

    def get_parameters(self):
        return { 'model': np.array(self._model) }

    def set_parameters(self, parameters):
        self._model = np.array(parameters['model'])

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
        action_logits = self._model.dot(observation).flatten()
        action_distribution = self._pd_factory.probability_distribution(action_logits)
        # print(self._model.shape, observation.shape, action_logits.shape)
        if deterministic:
            return action_distribution.mode()
        else:
            return action_distribution.sample()

    def step(self, observation, deterministic):
        observation = observation.reshape(-1, 1)
        action_logits = self._model.dot(observation).flatten()
        action_distribution = self._pd_factory.probability_distribution(action_logits)
        # print(self._model.shape, observation.shape, action_logits.shape)
        if deterministic:
            return action_distribution.mode(), action_distribution
        else:
            return action_distribution.sample(), action_distribution
