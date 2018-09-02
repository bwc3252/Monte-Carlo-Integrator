from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal


import mcsampler_new


# get user-provided parameters for the test
d = int(input('How many dimensions? '))
k = int(input('How many components? '))
llim, rlim = -40, 40
means = np.random.uniform(llim, rlim, size=(k, d))
weights = np.random.uniform(size=k)
weights /= np.sum(weights)

# print the randomly-generated components
for index in range(k):
    mean = means[index]
    weight = weights[index]
    print('Component', index)
    print('Mean:')
    print(mean)
    print('Mixture weight:', weight)

# integrand
def integrand(samples):
    ret = 0
    for index in range(k):
        w = weights[index]
        mean = means[index]
        ret += w * multivariate_normal.pdf(samples, mean)
    return np.rot90([ret], -1)


sampler = mcsampler_new.MCSampler()
args = []
for index in range(d):
    sampler.add_parameter(str(index), left_limit=llim-10, right_limit=rlim+10)
    args.append(str(index))
integral, var, eff_samp, _ = sampler.integrate(integrand, args=args, n_comp=k, write_to_file=True)
print('\nFinal result (should be at most 1):')
print(integral, 'with variance', var, 'and eff_samp', eff_samp)
