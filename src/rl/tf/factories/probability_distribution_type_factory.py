import rl

from gym.spaces import Box, Discrete
from rl.tf.distributions import CategoricalProbabilityDistributionType

class ProbabilityDistributionTypeFactory:

    def create_probability_distribution_type(self, space):
        if isinstance(space, Discrete):
            return CategoricalProbabilityDistributionType(space.n)
        else:
            raise NotImplementedError()
