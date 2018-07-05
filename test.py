from __future__ import print_function
import numpy as np
import monte_carlo_integrator as monte_carlo
import matplotlib.pyplot as plt


x_min, x_max = -15, 15 # left and right limits, default to same for each dimension
dim = 1 # number of dimensions
bounds = np.array([[x_min, x_max]] * dim) # create array of limits of integration
gmm_dict = {} # initialize dict used to group dimensions
n_comp = 2 # number of gaussian components for each dimension
for x in range(dim):
    gmm_dict[(x,)] = None # default to modeling each dimension separately

    # Each dictionary key is a tuple of integers representing sample array
    # indices. Each value is the mixture model object, which defaults to None.
    #
    # To initialize a 5-dimensional gaussian mixture model where dimensions
    # 0 and 3 are correlated, for example, use:
    #
    # gmm_dict = {(0, 3):None, (1,):None, (2,):None, (4,):None}
    #
    # The indices correspond with the columns of the sample array.


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


integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp)
result = integrator.integrate(func=integrand, err_thresh=0.01, max_count=30)
#sample_array = result['sample_array'][-1]
print('final integral:', result['integral'], 'with error', result['error'])
