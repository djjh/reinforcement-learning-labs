import numpy as np

def discount_cumsum(rewards, discount):
    out = np.zeros_like(rewards)
    n = len(rewards)
    for i in reversed(range(n)):
        out[i] = rewards[i] + (discount*out[i+1] if i+1 < n else 0)
    return out
