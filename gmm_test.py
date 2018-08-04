from __future__ import print_function
import numpy as np
import gaussian_mixture_model as GMM

mean1 = [0, 0]
mean2 = [5, 5]
cov = [[1, .5,], [.5, 2]]
#mean1 = [0]
#mean2 = [5]
#mean3 = [-5]
#mean4 = [10]
#cov = [[1]]

samples = np.empty((10000, 2))
samples[0:5000] = np.random.multivariate_normal(mean1, cov, 5000)
samples[5000:10001] = np.random.multivariate_normal(mean2, cov, 5000)
model = GMM.gmm(2)
model.fit(samples)
for x in range(19):
    samples[0:5000] = np.random.multivariate_normal(mean1, cov, 5000)
    samples[5000:10001] = np.random.multivariate_normal(mean2, cov, 5000)
    model.update(samples)

print('Results after 20 iterations:\n')
print('Means\n', model.means)
print('Covariances\n', model.covariances)
print('Weights\n', model.weights)
print('\n')

new_model = GMM.gmm(2)
new_model.fit(samples)

print('Results of fitting a new model using last sample array alone:\n')
print('Means\n', new_model.means)
print('Covariances\n', new_model.covariances)
print('Weights\n', new_model.weights)
