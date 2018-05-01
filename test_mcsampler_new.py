import mcsampler_new
import numpy as np


def integrand(sample_array):
    # An example integrand where each dimension is the same mixture of to gaussians
    value_array1 = np.square(sample_array) * -0.5
    value_array2 = np.square(sample_array + 2) * -0.5
    value_array1 = np.exp(value_array1)
    value_array2 = np.exp(value_array2)
    value_array1 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array1)
    value_array2 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array2)

    value_array = value_array1 + value_array2

    value_array = np.rot90([np.prod(value_array, axis=1)], -1)

    return value_array


sampler = mcsampler_new.MCSampler()
sampler.add_parameter('x', left_limit=-10, right_limit=10)
sampler.add_parameter('y', left_limit=-10, right_limit=10)
i, err, arr = sampler.integrate(integrand, args=('x', 'y'), n_comp=1)
print('integral:', i, 'with error', err)
