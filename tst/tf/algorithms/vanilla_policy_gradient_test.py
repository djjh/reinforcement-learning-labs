import pytest
import rl
import mock
from mock import Mock

from rl.tf.algorithms import VanillaPolicyGradient


class VanillaPolicyGradientTest:

    def test_init(self):
        with pytest.raises(TypeError):
            VanillaPolicyGradient()

        VanillaPolicyGradient(
            environment=Mock(),
            random_seed=None,
            policy_factory=Mock(),
            advantage_function=None,
            Rollout=None,
            min_steps_per_batch=None)

    def test_update(self):
        environment = Mock()
        random_seed = 0
        policy_factory = Mock()
        advantage_function = Mock()
        episode = Mock()
        episode.__len__ = Mock(return_value=2)
        episode.get_observations = Mock(return_value=[0, 1])
        episode.get_actions = Mock(return_value=[0, 1])
        Rollout = Mock(return_value=episode)
        min_steps_per_batch = 1

        vpg = VanillaPolicyGradient(
            environment,
            random_seed,
            policy_factory,
            advantage_function,
            Rollout,
            min_steps_per_batch)

        vpg.update()
