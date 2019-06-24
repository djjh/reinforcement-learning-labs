import gym
import numpy as np
import random
import scipy.special
import tensorflow as tf

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete
from rl import Framework

class InputFactory:

    def create(self, framework, space, batch_size):

        assert framework == Framework.TENSORFLOW, "Unsupported framework, try the delegating InputFactory"

        if isinstance(space, Box):
            return BoxInput(space, batch_size)
        else:
            raise NotImplementedError()

class BoxInput:

    def __init__(self, space, batch_size):
        self._space = space
        self._batch_size = batch_size

        self._shape = self._space.shape
        # self._flat_length = np.prod(self._space)

        self._placeholder = tf.placeholder(shape=(batch_size,) + self._shape, dtype=space.dtype)
        self._input = tf.cast(self._placeholder, tf.float32)
        # TODO: scale like open ai baselines?

    def get_placeholder(self):
        return self._placeholder

    def get_input(self):
        return self._input

    # def get_flat_length(self):
    #     return self._flat_length
