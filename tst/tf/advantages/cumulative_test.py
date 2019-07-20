import rl

from rl.tf.advantages import Cumulative
from tst.utilities import generate_episodes


class CumulativeTest:

    def test_init(self):
        Cumulative()

    def test_works(self):
        advantage_function = Cumulative()
        episodes = generate_episodes([[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0], ])

        advantages = advantage_function.get_advantages(episodes)

        assert advantages == [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0]

    def test_update(self):
        advantage_function = Cumulative()
        episodes = generate_episodes([[0, 0, 0], [0, 1, 2], [2, 1, 0], [1, 1, 1]])

        advantage_function.update(episodes)

        # no exception thrown
