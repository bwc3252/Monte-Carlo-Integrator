from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

'''
Code adapted from http://www.aishack.in/tutorials/generating-multivariate-gaussian-random/
'''


def get_corner_coords(bounds):
    d = len(bounds)
    points = np.empty((2**d, d))
    bounds = np.rot90(bounds)
    i = np.matrix(list(itertools.product([0,1], repeat=d)))
    index = 0
    for slice in i:
        t = np.diag(bounds[slice][0])
        points[index] = t
        index += 1
    return points


def get_new_bounds(bounds, q):
    r = q.I # inverse of transformation
    d = len(bounds)
    old_points = get_corner_coords(bounds)
    new_points = np.empty((2**d, d))
    index = 0
    for point in old_points:
        new = np.dot(r, point)
        new_points[index] = new
        index += 1
    new_bounds = np.empty((d, 2))
    for dim in range(d):
        new_bounds[dim][0] = min(new_points[:,[dim]])
        new_bounds[dim][1] = max(new_points[:,[dim]])
    return new_bounds


def get_multipliers(cov):
    [lam, sigma] = np.linalg.eig(cov)
    lam = np.matrix(np.diag(np.sqrt(lam)))
    q = np.matrix(sigma) * lam
    return q


def get_one_sample(bounds, new_bounds, mean, q):
    d = len(bounds)
    count = 0
    llim = new_bounds[:,[0]]
    rlim = new_bounds[:,[1]]
    while True:
        count += 1
        sample = np.rot90(truncnorm.rvs(llim, rlim, loc=0, scale=1))
        sample = np.dot(q, sample[0]) + mean # unrotated here
        if sample_in_region(sample, bounds):
            return sample, count


def sample_in_region(sample, bounds):
    d = len(bounds)
    sample = np.array(sample)[0]
    for dim in range(d):
        temp = np.array(bounds[dim])
        llim = temp[0][0]
        rlim = temp[0][1]
        if sample[dim] < llim or sample[dim] > rlim:
            return False
    return True


def sample(mean, cov, bounds, n):
    mean = np.matrix(mean)
    d = len(bounds)
    q = get_multipliers(cov)
    new_bounds = get_new_bounds(bounds - mean, q)
    llim_new = new_bounds[:,[0]]
    rlim_new = new_bounds[:,[1]]
    ret = np.empty((0, d))
    while len(ret) < n:
        samples = np.rot90(truncnorm.rvs(llim_new, rlim_new, loc=0, scale=1, size=(d, n)))
        samples = np.rot90(np.inner(q, samples)) + mean
        llim = np.rot90(bounds[:,[0]])
        rlim = np.rot90(bounds[:,[1]])
        replace1 = np.greater(samples, llim).all(axis=1)
        replace2 = np.less(samples, rlim).all(axis=1)
        replace = np.array(np.logical_and(replace1, replace2)).flatten()
        to_append = samples[replace]
        ret = np.append(ret, to_append, axis=0)
    return ret[:n]
