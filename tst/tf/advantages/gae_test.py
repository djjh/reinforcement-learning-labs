import pytest
import rl
import mock
from mock import Mock

from rl.tf.advantages import Gae
from rl.tf.values import ValueFunction
from tst.utilities import generate_episodes


import numpy as np

class GaeTest:

    def test_init(self):
        with pytest.raises(TypeError):
            Gae()

        with pytest.raises(TypeError):
            Gae(value_function=None, gamma=None)

        with pytest.raises(TypeError):
            Gae(value_function=None, lambduh=None)

        with pytest.raises(TypeError):
            Gae(gamma=None, lambduh=None)

        Gae(value_function=None, gamma=None, lambduh=None)

    @pytest.mark.parametrize("gamma, lambduh, rewards, expected_advantages", [
        pytest.param(
            1,
            1,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 3, 3, 2, 3, 1, 0, 3, 2, 1, 0, 0, 0],
            id='full-lambda,full-gamma => (same as discount_cumsum)'
        ),
        pytest.param(
            1,
            0.5,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [ 0, 0, 0]],
            [0, 0, 0, 1, 2, 2, 2.5, 1, 0, 1.75, 1.5, 1, 0, 0, 0],
            id='full-lambda,half-gamma => (same as discount_cumsum)'
        ),
        pytest.param(
            1,
            0,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='full-lambda,zero-gamma => (same as discount_cumsum)'
        ),
        pytest.param(
            0.5,
            1,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 1, 2, 2, 2.5, 1, 0, 1.75, 1.5, 1, 0, 0, 0],
            id='half-lambda,full-gamma => (same as full-lambda,half-gamma)'
        ),
        pytest.param(
            0.5,
            0.5,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0.375, 1.5, 2, 2.25, 1, 0, 1.3125, 1.25, 1, 0, 0, 0],
            id='half-lambda,half-gamma'
        ),
        pytest.param(
            0.5,
            0,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='half-lambda,zero-gamma => (same as zero-lambda,half-gamma)'
        ),
        pytest.param(
            0,
            1,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='zero-lambda,zero-gamma => (same as zero-lambda,zero-gamma)'
        ),
        pytest.param(
            0,
            0.5,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='zero-lambda,half-gamma => (same as zero-lambda,zero-gamma)'
        ),
        pytest.param(
            0,
            0,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='zero-lambda,zero-gamma'
        ),
    ])
    def test_works(self, gamma, lambduh, rewards, expected_advantages):
        value_function = Mock()
        value_function.__enter__ = Mock(return_value=(Mock(), None))
        value_function.__exit__ = Mock(return_value=None)
        value_function.get_values = Mock()
        value_function.get_values.side_effect = np.asarray(rewards)


        advantage_function = Gae(value_function=value_function, gamma=gamma, lambduh=lambduh)
        episodes = generate_episodes(rewards)

        advantages = advantage_function.get_advantages(episodes)

        assert advantages == expected_advantages

    def test_update(self):
        value_function = Mock()
        value_function.__enter__ = Mock(return_value=(Mock(), None))
        value_function.__exit__ = Mock(return_value=None)

        advantage_function = Gae(value_function=value_function, gamma=1, lambduh=1)
        episodes = generate_episodes([[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1]])

        advantage_function.update(episodes)

        # no exception thrown


class ValueFunctionImpl(ValueFunction):

    def __init__(self):
        pass
