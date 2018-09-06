from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
import sys


import mcsampler_new


# integrand
def integrand(samples):
    ret = 0
    for index in range(k):
        w = weights[index]
        mean = means[index]
        ret += w * multivariate_normal.pdf(samples, mean)
    return np.rot90([ret], -1)


def user_func1(integrator):
    print()


def user_func2(sampler, integrator):
    print('This should print at the end')
    print('Means of gmm:')
    result = integrator.gmm_dict
    for index in result:
        print(result[index].means)


if __name__ == '__main__':
    # check for valid command line arguments
    if len(sys.argv) != 4:
        print('Usage: python/python3 comprehensive_test.py [ndim] [ncomp] [same model (y/n)]')
    else:
        d = int(sys.argv[1])
        k = int(sys.argv[2])
        group = sys.argv[3].lower()
        if group == 'n':
            s = 'not'
        else:
            s = ''
        print('Running test with', k, 'components in', d, 'dimensions where dimensions are',
                s, 'modeled together')
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

        sampler = mcsampler_new.MCSampler()
        args = []
        for index in range(d):
            sampler.add_parameter(str(index), left_limit=llim-10, right_limit=rlim+10)
            args.append(str(index))
        if group == 'y':
            gmm_dict = {}
            gmm_dict[tuple(range(d))] = None
        else:
            gmm_dict = None
        integral, var, eff_samp, _ = sampler.integrate(integrand, args=args, n_comp=k,
                                    write_to_file=True, gmm_dict=gmm_dict, mcsamp_func=user_func2,
                                    integrator_func=user_func1)
        print('\nFinal result (should be about 1, unless a Gaussian is close to a boundary):')
        print(integral, 'with variance', var, 'and eff_samp', eff_samp)
