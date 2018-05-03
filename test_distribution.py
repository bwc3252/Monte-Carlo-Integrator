import mcsampler_new
import numpy as np
import scipy.stats as stats

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
result_dict = sampler.integrate(integrand, args=('x',), n_comp=2) # do a 1-d integral
value_array = result_dict['value_array'][-1]
sample_array = result_dict['sample_array'][-1]
integral = result_dict['integral']
err = result_dict['error']
print('integral:', integral, 'with error', err)

##############################################################
#                                                            #
#    Use last round of samples and values to compare CDFs    #
#                                                            #
##############################################################

# create actual, ideal CDF

samples = np.linspace(-12, 12, 1000)
cdf = (stats.norm.cdf(samples, loc=-4, scale=1) + stats.norm.cdf(samples, loc=4, scale=1)) / 2

# create CDF from data

to_plot = np.zeros((len(value_array), 2))
to_plot[:,[0]] = sample_array
to_plot[:,[1]] = value_array
to_plot = np.sort(to_plot, axis=0) # sort x values
cumsum = np.cumsum(to_plot[:,[1]])
to_plot[:,[1]] = np.rot90([cumsum], -1) / cumsum[-1] # rotate and normalize

# plot the CDFs

plt.plot(samples, cdf, color='red', label='Actual CDF')
plt.plot(to_plot[:,[0]], to_plot[:,[1]], color='green', label='Sampled CDF')
plt.legend()
plt.show()
