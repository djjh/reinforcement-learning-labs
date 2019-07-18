
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete


class InputFactory:

    def create(self, space, batch_size):

        if isinstance(space, Box):
            return BoxInput(space, batch_size)
        else:
            raise NotImplementedError()
