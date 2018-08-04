import numpy as np
import gaussian_mixture_model as GMM
import matplotlib.pyplot as plt


def f(sample_array):
    # An example integrand where each dimension is the same mixture of to gaussians
    value_array1 = np.square(sample_array - 4) * -0.5
    value_array2 = np.square(sample_array + 4) * -0.5
    value_array1 = np.exp(value_array1)
    value_array2 = np.exp(value_array2)
    value_array1 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array1)
    value_array2 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array2)

    value_array = value_array1 + value_array2

    value_array = np.rot90([np.prod(value_array, axis=1)], -1)

    return value_array

n = 1000
model = GMM.gmm(2)
sample_array = np.random.uniform(-15, 15, (n, 1))
value_array = f(sample_array)
model.fit(sample_array, sample_weights=value_array)
for iteration in range(20):
    sample_array = model.sample(n)
    value_array = f(sample_array)
    p_array = model.score(sample_array)
    weights = value_array / p_array
    model.update(sample_array, sample_weights=weights)
print(model.means)
plt.hist(sample_array, 100)
plt.show()
