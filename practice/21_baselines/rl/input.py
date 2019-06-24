import gym
import numpy as np
import random
import scipy.special

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete
import rl
from rl import Framework

class InputFactory:

    def create(self, framework, space, batch_size):
        if framework == Framework.TENSORFLOW:
            input_factory = rl.tf.InputFactory()
        else:
            raise NotImplementedError()

        return input_factory.create(
            framework=framework,
            space=space,
            batch_size=batch_size)
