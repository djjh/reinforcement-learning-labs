from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiBinary
from gym.spaces import MultiDiscrete
from rl.tf.input import BoxInput


class InputFactory:

    def create(self, space, batch_size):
        if isinstance(space, Box):
            return BoxInput(space, batch_size)
        else:
            raise NotImplementedError()
