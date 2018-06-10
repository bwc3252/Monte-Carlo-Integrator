from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import time

import multivariate_truncnorm as truncnorm

bounds = np.matrix([[0, 3], [0, 5]])
cov = np.matrix([[1, .8], [.8, 2]])
mean = np.array([0, 0])
n = 50000

t1 = time()
samples1= truncnorm.sample(mean, cov, bounds, n)
t2 = time()

t3 = time()
samples2 = np.random.multivariate_normal(mean, cov, n)
t4 = time()

print('truncnorm time:', t2 - t1, 'seconds')
print('numpy time:', t4 - t3, 'seconds')

plt.figure(1)
plt.subplot(221)
plt.axis([-5, 5, -5, 5])
#print( samples1.T)
x=np.ravel((samples1.T)[0]); y=np.ravel(samples1.T[1]);
#print(x,y)
plt.scatter(x,y)

plt.subplot(222)
plt.axis([-5, 5, -5, 5])
plt.scatter(samples2[:,[0]], samples2[:,[1]])

plt.subplot(223)
plt.hist(samples1[:,[0]], 50)

plt.subplot(224)
plt.hist(samples1[:,[1]], 50)
#plt.show()
plt.savefig("dat.png")


def mycdf(dat):
    def counter(x):
        if isinstance(x,float):
            return np.sum(dat < x)*1.0/len(dat)
        else:
            yvals = np.zeros(len(x))
            for indx in np.arange(len(yvals)):
                yvals[indx] = np.sum(dat < x[indx])*1.0/len(dat)
            return yvals
    return counter

# Apply constraints on multivariate normal
indx_ok = np.logical_and(samples2[:,0]>bounds[0,0]  , samples2[:,0]<bounds[0,1] )
indx_ok = np.logical_and(indx_ok, samples2[:,1]>bounds[1,0])
indx_ok = np.logical_and(indx_ok, samples2[:,1]<bounds[1,1])
samples2 = samples2[indx_ok]

# Confirm CDF is consistent
fn = mycdf(samples2[:,0])
xvals = np.linspace(bounds[0,0], bounds[0,1],100)
yvals = fn(xvals)
plt.clf()
plt.plot(xvals,yvals)
fn = mycdf(samples1[:,0])
xvals = np.linspace(bounds[0,0], bounds[0,1],100)
yvals = fn(xvals)
plt.plot(xvals,yvals)
plt.savefig("cdf.png")
