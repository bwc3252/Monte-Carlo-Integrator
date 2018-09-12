import numpy as np
import matplotlib.pyplot as plt
import gaussian_mixture_model as GMM


def get_data(d, k):
    means = [0, 5]
    data = np.empty((0, 2))
    for n in range(1000, 10001, 1000):
        samples = np.empty((n, d))
        for comp in range(k):
            start = (comp / k) * n
            end = ((comp + 1) / k) * n
            samples[int(start):int(end)] = np.random.normal(means[comp], 1, (int(end - start), d))
        model = GMM.gmm(k)
        model.fit(samples)
        row = np.array([[n, abs(model.log_prob)]])
        data = np.append(data, row, axis=0)
    return data

data11 = get_data(1, 1)
data12 = get_data(1, 2)
data21 = get_data(2, 1)
data22 = get_data(2, 2)

plt.figure(1)

plt.subplot(221)
plt.plot(data11[:,[0]], data11[:,[1]])
plt.xlabel('number of data points')
plt.ylabel('abs(log_prob) after training')
plt.title('dim = 1, n_comp = 1')

plt.subplot(222)
plt.plot(data12[:,[0]], data12[:,[1]])
plt.xlabel('number of data points')
plt.ylabel('abs(log_prob) after training')
plt.title('dim = 1, n_comp = 2')

plt.subplot(223)
plt.plot(data21[:,[0]], data21[:,[1]])
plt.xlabel('number of data points')
plt.ylabel('abs(log_prob) after training')
plt.title('dim = 2, n_comp = 1')

plt.subplot(224)
plt.plot(data22[:,[0]], data22[:,[1]])
plt.xlabel('number of data points')
plt.ylabel('abs(log_prob) after training')
plt.title('dim = 2, n_comp = 2')

plt.show()
