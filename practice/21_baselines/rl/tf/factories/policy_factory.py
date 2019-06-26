
from rl.tf.policies import LinearPolicy

class PolicyFactory:

    def __init__(self, input_factory, distribution_type_factory):
        self._input_factory = input_factory
        self._distribution_type_factory = distribution_type_factory

    def create_policy(self, observation_space, action_space, session):
        return LinearPolicy(
            observation_space=observation_space,
            action_space=action_space,
            input_factory=self._input_factory,
            distribution_type_factory=self._distribution_type_factory,
            session=session)
