import pytest
import rl

from rl.tf.advantages import RewardToGo
from tst.utilities import generate_episodes


class RewardToGoTest:

    def test_init(self):
        with pytest.raises(TypeError):
            RewardToGo()

        RewardToGo(discount=0.0)
        RewardToGo(discount=0.5)
        RewardToGo(discount=1.0)
        RewardToGo(discount=None)

    @pytest.mark.parametrize("discount, rewards, expected_advantages", [
        pytest.param(
            1,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 3, 3, 2, 3, 1, 0, 3, 2, 1, 0, 0, 0],
            id='full-discount'
        ),
        pytest.param(
            0.5,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [ 0, 0, 0]],
            [0, 0, 0, 1, 2, 2, 2.5, 1, 0, 1.75, 1.5, 1, 0, 0, 0],
            id='half-discount'
        ),
        pytest.param(
            0,
            [[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            [0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0],
            id='no-discount'
        )
    ])
    def test_works(self, discount, rewards, expected_advantages):
        advantage_function = RewardToGo(discount=discount)
        episodes = generate_episodes(rewards)

        advantages = advantage_function.get_advantages(episodes)

        assert advantages == expected_advantages

    def test_update(self):
        advantage_function = RewardToGo(discount=1)
        episodes = generate_episodes([[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1]])

        advantage_function.update(episodes)

        # no exception thrown
