import numpy as np
import matplotlib.pyplot as pp
import scipy.stats




mean = 0      # Aka mu.
variance = 1  # Aka sigma squared.
standard_deviation = np.sqrt(variance) # Aka sigma.


# Sample from gaussian distribution.

num_samples = 1000
num_bins = 30

samples = np.random.standard_normal(num_samples)


# Plot gaussian distribution curve.

x = np.arange(samples.min(), samples.max(), 0.001)
y = scipy.stats.norm.pdf(x, mean, variance)


pp.hist(samples, bins=num_bins, density=True)
pp.plot(x, y)
pp.show()
