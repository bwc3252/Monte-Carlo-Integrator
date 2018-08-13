from __future__ import print_function
from sklearn.externals import joblib
import numpy as np
import mcsampler_new
import mcsampler


gp = joblib.load('gp_fit.pkl')


def fit_func(mass_coord1, mass_coord2):
    return gp.predict([[mass_coord1, mass_coord2]])#[0]


def integrand(sample_array):
    length = len(sample_array)
    index = 0
    value_array = np.empty((length, 1))
    while index < length:
        sample = sample_array[index]
        value = fit_func(sample[0], sample[1])
        value_array[(index, 0)] = value
        index += 1
    return value_array


def main():

    # get results for existing integrator

    print('OLD INTEGRATOR')
    existing = mcsampler.MCSampler()
    existing.add_parameter('mass_coord1', np.vectorize(lambda x: 1.3), left_limit=1.217, right_limit=1.219)
    existing.add_parameter('mass_coord2', np.vectorize(lambda x: 1.3), left_limit=0.24, right_limit=0.249999)
    result = existing.integrate(np.vectorize(fit_func), 'mass_coord1', 'mass_coord2', nmax=1e4)
    print(result, '\n')
    print('_hist')
    print(existing._hist)

    # get results for new integrator

    print('NEW INTEGRATOR')
    sampler = mcsampler_new.MCSampler()
    sampler.add_parameter('mass_coord1', left_limit=1.217, right_limit=1.219)
    sampler.add_parameter('mass_coord2', left_limit=0.24, right_limit=0.249999)
    integral, var, eff_samp, _ = sampler.integrate(integrand, args=('mass_coord1', 'mass_coord2'), n_comp=2, write_to_file=True, gmm_dict={(0, 1):None})
    print('integral:', integral, 'variance:', var, 'eff_samp:', eff_samp)


if __name__ == '__main__':
    main()
