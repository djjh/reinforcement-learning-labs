import nevergrad as ng
import rl

from rl.core import Rollout
from rl.np.algorithms import Nevergrad
from rl.np.algorithms import RandomSearch
from rl.np.algorithms import UniformRandom
from rl.np.algorithms import VanillaPolicyGradient
from rl.np.factories import ProbabilityDistributionTypeFactory
from rl.np.factories import InputFactory
from rl.np.factories import PolicyFactory


def create_linear_policy_factory():
    return PolicyFactory(
        input_factory=InputFactory(),
        distribution_type_factory=ProbabilityDistributionTypeFactory())

def create_scratch_vpg_v0(environment, random_seed):
    return VanillaPolicyGradient(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        rollout_factory=Rollout,
        min_steps_per_batch=1)

def create_scratch_random_search_v0(environment, random_seed):
    return RandomSearch(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=Rollout,
        batch_size=1,
        explore=0.5)

def create_scratch_uniform_random_v0(environment, random_seed):
    return UniformRandom(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=Rollout,
        batch_size=1,
        low=-1.0,
        high=1.0)

def create_scratch_one_plus_one_v0(environment, random_seed):
    return Nevergrad(
        environment=environment,
        random_seed=random_seed,
        policy_factory=create_linear_policy_factory(),
        create_rollout=Rollout,
        optimizer=ng.optimizers.OnePlusOne,
        budget=400,
        low=-1.0,
        high=1.0)


class AlgorithmFactory:

    def __init__(self):
        self._algorithms = {
            'Scratch-UniformRandom-v0': create_scratch_uniform_random_v0,
            'Scratch-RandomSearch-v0': create_scratch_random_search_v0,
            'Scratch-OnePlusOne-v0': create_scratch_one_plus_one_v0,
            'Scratch-VanillaPolicyGradient-v0': create_scratch_vpg_v0
        }

    def get_algorithms(self):
        return self._algorithms

    def create_algorithm(self, algorithm_name, environment, random_seed):
        return self._algorithms[algorithm_name](environment, random_seed)
