import numpy as np
import gaussian_mixture_model as GMM
import matplotlib.pyplot as plt


def f(sample_array):
    value_array1 = np.square(sample_array - 4) * -0.5
    value_array2 = np.square(sample_array + 4) * -0.5
    value_array1 = np.exp(value_array1)
    value_array2 = np.exp(value_array2)
    value_array1 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array1)
    value_array2 = np.multiply((1 / (np.sqrt(2 * np.pi))), value_array2)

    value_array = value_array1 + 2 * value_array2

    value_array = np.rot90([np.prod(value_array, axis=1)], -1)

    return value_array


model = GMM.gmm(2)
sample_array = np.random.uniform(-10, 10, (1000, 1))
value_array = f(sample_array)
model.fit(sample_array, sample_weights=value_array)
print('Initial fit:')
print(model.means)
print(model.covariances)
print(model.weights)

scores = model.score(sample_array)
plt.scatter(sample_array, scores)
plt.show()

for iteration in range(19):
    sample_array = np.random.uniform(-15, 15, (1000, 1))
    value_array = f(sample_array)
    model.update(sample_array, value_array)

print('20 iterations:')
print(model.means)
print(model.covariances)
print(model.weights)

scores = model.score(sample_array)
plt.scatter(sample_array, scores)
plt.show()
