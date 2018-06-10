from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import time

import multivariate_truncnorm as truncnorm

bounds = np.matrix([[-5, 5], [0, 5]])
cov = np.matrix([[.1, .3], [.3, 1]])
mean = np.array([0, 0])
n = 5000

t1 = time()
samples1, ratio = truncnorm.sample(mean, cov, bounds, n, ratio=True)
t2 = time()

t3 = time()
samples2 = np.random.multivariate_normal(mean, cov, n)
t4 = time()

print('truncnorm time:', t2 - t1, 'seconds, with acceptance ratio of', ratio)
print('numpy time:', t4 - t3, 'seconds')

plt.figure(1)
plt.subplot(211)
plt.axis([-5, 5, -5, 5])
plt.scatter(samples1[:,[0]], samples1[:,[1]])

plt.figure(1)
plt.subplot(212)
plt.axis([-5, 5, -5, 5])
plt.scatter(samples2[:,[0]], samples2[:,[1]])
plt.show()

plt.hist(samples1[:,[0]], 50)
plt.show()

plt.hist(samples1[:,[1]], 50)
plt.show()
