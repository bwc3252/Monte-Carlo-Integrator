from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal


import mcsampler_new


# get user-provided parameters for the test
d = int(input('How many dimensions? '))
k = int(input('How many components? '))
group = input('Should dimensions be grouped together (y/n): ').lower()
llim, rlim = -10, 10
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


def user_func1(integrator):
    print('This should print every iteration')


def user_func2(sampler, integrator):
    print('This should print at the end')
    print('Means of gmm:')
    result = integrator.gmm_dict
    for index in result:
        print(result[index].means)


sampler = mcsampler_new.MCSampler()
args = []
for index in range(d):
    sampler.add_parameter(str(index), left_limit=llim-10, right_limit=rlim+10)
    args.append(str(index))
if group == 'y':
    gmm_dict = {}
    gmm_dict[range(d)] = None
else:
    gmm_dict = None
integral, var, eff_samp, _ = sampler.integrate(integrand, args=args, n_comp=k,
                            write_to_file=True, gmm_dict=gmm_dict, mcsamp_func=user_func2,
                            integrator_func=user_func1)
print('\nFinal result (should be about 1, unless a Gaussian is close to a boundary):')
print(integral, 'with variance', var, 'and eff_samp', eff_samp)
