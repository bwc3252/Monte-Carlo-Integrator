import numpy as np
import gaussian_mixture_model as GMM
import multivariate_truncnorm as truncnorm


mean1, mean2 = np.array([[0]]), np.array([[5]])
cov = np.array([[1]])
n = 10000
bounds = np.array([[0, 5]])

sample_array = np.empty((10000, 1))
sample_array[0:5000] = truncnorm.sample(mean1, cov, bounds, 5000)
sample_array[5000:10001] = truncnorm.sample(mean2, cov, bounds, 5000)

model = GMM.gmm(2)
model.fit(sample_array, bounds=bounds, trunc_corr=True)
model.print_params()
