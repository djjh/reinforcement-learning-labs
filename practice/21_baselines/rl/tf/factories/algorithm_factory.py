import rl

from rl.core import Rollout
from rl.tf.algorithms import VanillaPolicyGradient
from rl.tf.factories import ProbabilityDistributionTypeFactory
from rl.tf.factories import InputFactory
from rl.tf.factories import PolicyFactory

##########
# Common #
##########

def create_policy_factory():
    return PolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())


##############
# Algorithms #
##############

def create_tensorflow_vpg_v0(environment, random_seed):
    return VanillaPolicyGradient(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_policy_factory(),
        Rollout=Rollout,
        min_steps_per_batch=1)


######################
# Algorithms Factory #
######################

class AlgorithmFactory:

    def __init__(self):
        self._algorithms = {
            'Tensorflow-VanillaPolicyGradient-v0': create_tensorflow_vpg_v0
        }

    def get_algorithms(self):
        return self._algorithms

    def create_algorithm(self, algorithm_name, environment, random_seed):
        return self._algorithms[algorithm_name](environment, random_seed)
