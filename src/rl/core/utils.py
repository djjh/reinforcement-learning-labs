import numpy as np
import itertools

def discount_cumsum(rewards, discount):
    return reversed(list(itertools.accumulate(reversed(rewards), lambda a, b: discount * a + b)))
