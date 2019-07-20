import pytest
import rl

from rl.tf.advantages import AdvantageFunction
from tst.utilities import generate_episodes


class AdvantageFunctionTest:

    def test_init(self):
        with pytest.raises(NotImplementedError):
            AdvantageFunction()

    def test_advantages(self):
        advantage_function = AdvantageFunctionImpl()
        episodes = generate_episodes([[0]])

        with pytest.raises(NotImplementedError):
            advantage_function.get_advantages(episodes)

    def test_update(self):
        advantage_function = AdvantageFunctionImpl()
        episodes = generate_episodes([[0]])

        with pytest.raises(NotImplementedError):
            advantage_function.update(episodes)


class AdvantageFunctionImpl(AdvantageFunction):

    def __init__(self):
        pass
