import rl


from rl.core import Rollout
from rl.core import generate_functions
from rl.tf.algorithms import VanillaPolicyGradient
from rl.tf.factories import ProbabilityDistributionTypeFactory
from rl.tf.factories import InputFactory
from rl.tf.factories import PolicyFactory



############
# Policies #
############

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
        min_steps_per_batch=1,
        learning_rate=1e-2)
    #
    # args = {
    #     VanillaPolicyGradient: {
    #         'environment': [environment],
    #         'random_seed': [random_seed],
    #         'policy_factory': [
    #             {
    #                 PolicyFactory: {
    #                     'input_factory': [{ InputFactory: {} }],
    #                     'distribution_type_factory': [{ ProbabilityDistributionTypeFactory: {} }]
    #                 }
    #             }
    #         ],
    #         'Rollout': [Rollout],
    #         'min_steps_per_batch': [1],
    #         'learning_rate': [1e-2, 1e-3]
    #     },
    # }
    #
    # return next(generate_functions(args))()


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


    def generate_algorithms(self):
        for f in self.generate_algorithm_functions():
            yield f()


    def generate_algorithm_functions(self):
        args = {
            VanillaPolicyGradient: {
                'environment': [environment],
                'random_seed': [random_seed],
                'policy_factory': [
                    {
                        PolicyFactory: {
                            'input_factory': [{ InputFactory: {} }],
                            'distribution_type_factory': [{ ProbabilityDistributionTypeFactory: {} }]
                        }
                    }
                ],
                'Rollout': [Rollout],
                'min_steps_per_batch': [1],
                'learning_rate': [1e-2, 1e-3]
            },
        }

        yield from generate_functions(args)
