from __future__ import print_function
import mcsampler_new
import numpy as np

#import matplotlib.pyplot as plt


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
integral, var, eff_samp, _ = sampler.integrate(integrand, args=('x',), n_comp=2, write_to_file=True, var_thresh=0.02) # do a 1-d integral
print('\nFinal result:')
print(integral, 'with variance', var, 'and eff_samp', eff_samp)
