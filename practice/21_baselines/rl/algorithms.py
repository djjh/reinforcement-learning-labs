import nevergrad as ng

import rl
import rl.np
import rl.np.algorithms
import rl.tf
import rl.tf.algorithms

from rl import ProbabilityDistributionTypeFactory
from rl import InputFactory
from rl import LinearPolicyFactory
from rl import rollout


##########
# Common #
##########

def create_linear_policy_factory():
    return LinearPolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())


##############
# Tensorflow #
##############

def create_tensorflow_vpg_v0(environment, random_seed):
    return rl.tf.algorithms.VanillaPolicyGradient(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        rollout_factory=rollout,
        min_steps_per_batch=1)


###########
# Scratch #
###########

def create_scratch_vpg_v0(environment, random_seed):
    return rl.np.algorithms.VanillaPolicyGradient(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        rollout_factory=rollout,
        min_steps_per_batch=1)

def create_scratch_random_search_v0(environment, random_seed):
    return rl.np.algorithms.RandomSearch(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=rollout,
        batch_size=1,
        explore=0.5)

def create_scratch_uniform_random_v0(environment, random_seed):
    return rl.np.algorithms.UniformRandom(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=rollout,
        batch_size=1,
        low=-1.0,
        high=1.0)

def create_scratch_one_plus_one_v0(environment, random_seed):
    return rl.np.algorithms.Nevergrad(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=rollout,
        optimizer=ng.optimizers.OnePlusOne,
        budget=400,
        low=-1.0,
        high=1.0)


class AlgorithmFactory:

    def __init__(self):
        self._algorithms = {
            'Tensorflow-VanillaPolicyGradient-v0': create_tensorflow_vpg_v0,
            'Scratch-UniformRandom-v0': create_scratch_uniform_random_v0,
            'Scratch-RandomSearch-v0': create_scratch_random_search_v0,
            'Scratch-OnePlusOne-v0': create_scratch_one_plus_one_v0,
            'Scratch-VanillaPolicyGradient-v0': create_scratch_vpg_v0
        }

    def create_algorithm(self, algorithm_name, environment, random_seed):
        return self._algorithms[algorithm_name](environment, random_seed)
