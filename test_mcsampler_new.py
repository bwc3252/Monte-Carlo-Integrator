import mcsampler_new
import numpy as np

import matplotlib.pyplot as plt


def integrand(sample_array):
    # An example integrand where each dimension is the same mixture of to gaussians
    value_array1 = np.square(sample_array - 4) * -0.5
    value_array2 = np.square(sample_array + 4) * -0.5
    value_array1 = np.exp(value_array1)
    value_array2 = np.exp(value_array2)
    value_array1 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array1)
    value_array2 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array2)

    value_array = value_array1 + value_array2

    value_array = np.rot90([np.prod(value_array, axis=1)], -1)

    return value_array


sampler = mcsampler_new.MCSampler()
sampler.add_parameter('x', left_limit=-12, right_limit=12)
result_dict = sampler.integrate(integrand, args=('x',), n_comp=2, write_to_file=True)[3] # do a 1-d integral
value_array = result_dict['value_array'][-1]
p_array = result_dict['p_array'][-1]
sample_array = result_dict['sample_array'][-1]
value_array *= p_array
integral = result_dict['integral']
err = result_dict['error']
print('integral:', integral, 'with error', err)
