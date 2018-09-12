'''
test file for monte_carlo_integrator.py itself (without mcsampler)
'''

from __future__ import print_function
import numpy as np
import monte_carlo_integrator as monte_carlo



x_min, x_max = -15, 15 # left and right limits, default to same for each dimension
dim = 1 # number of dimensions
bounds = np.array([[x_min, x_max]] * dim) # create array of limits of integration
gmm_dict = {} # initialize dict used to group dimensions
n_comp = 2 # number of gaussian components for each independent GMM
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
#gmm_dict = {(0, 1):None}


def integrand(sample_array):
    # An example integrand where each dimension is the same mixture of two gaussians
    value_array1 = np.square(sample_array - 4) * -0.5
    value_array2 = np.square(sample_array + 4) * -0.5
    value_array1 = np.exp(value_array1)
    value_array2 = np.exp(value_array2)
    value_array1 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array1)
    value_array2 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array2)

    value_array = value_array1 + value_array2

    value_array = np.rot90([np.prod(value_array, axis=1)], -1)

    return value_array


def print_func(integrator):
    i = integrator.integral
    var = integrator.var
    eff_samp = integrator.eff_samp
    print('Integral:', i, 'variance:', var, 'eff_samp:', eff_samp, '\n')


# initialize the integrator with the correct parameters
integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, user_func=print_func)
# integrate the function
integrator.integrate(integrand, var_thresh=0.0001)
