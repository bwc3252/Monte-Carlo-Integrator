from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import multivariate_truncnorm as truncnorm
import itertools

'''
Equation references are from Numerical Recipes for general GMM and https://www.cs.nmsu.edu/~joemsong/publications/Song-SPIE2005-updated.pdf
for online updating features
'''

class estimator:

    def __init__(self, k, tol=0.001, max_iters=100):
        self.k = k
        self.tol = tol
        self.max_iters = max_iters
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.d = None
        self.p_nk = None
        self.log_prob = None

    def initialize(self, n, sample_array):
        self.means = sample_array[np.random.choice(n, self.k), :]
        self.covariances = [np.eye(self.d, self.d)] * self.k
        self.weights = np.array([1.0 / self.k] * self.k)

    def e_step(self, n, sample_array, sample_weights):
        if sample_weights is None:
            log_sample_weights = np.zeros((n, 1))
        else:
            log_sample_weights = np.log(sample_weights)
        p_nk = np.empty((n, self.k))
        for index in range(self.k):
            mean = self.means[index]
            cov = self.covariances[index]
            log_p = np.log(self.weights[index])
            log_pdf = np.rot90([multivariate_normal.logpdf(x=sample_array, mean=mean, cov=cov)], -1) # (16.1.4)
            p_nk[:,[index]] = log_pdf + log_p # (16.1.5)
        p_xn = logsumexp(p_nk, axis=1, keepdims=True) # (16.1.3)
        self.p_nk = p_nk - p_xn # (16.1.5)
        self.p_nk += log_sample_weights
        self.log_prob = np.sum(p_xn + log_sample_weights)# np.sum(np.log(p_xn * sample_weights)) # (16.1.2)

    def m_step(self, n, sample_array):
        p_nk = np.exp(self.p_nk)
        weights = np.sum(p_nk, axis=0)
        for index in range(self.k):
            # (16.1.6)
            w = weights[index]
            p_k = p_nk[:,[index]]
            mean = np.sum(np.multiply(sample_array, p_k), axis=0)
            mean /= w
            self.means[index] = mean
            # (16.1.6)
            diff = sample_array - mean
            cov = np.dot((p_k * diff).T, diff) / w
            self.covariances[index] = cov
            # (16.17)
        weights /= np.sum(p_nk)
        self.weights = weights

    def fit(self, sample_array, sample_weights):
        n, self.d = sample_array.shape
        self.initialize(n, sample_array)
        prev_log_prob = 0
        self.log_prob = float('inf')
        count = 0
        while abs(self.log_prob - prev_log_prob) > self.tol and count < self.max_iters: # abs(log_prob - prev_log_prob) > self.tol and
            #self.print_params()
            prev_log_prob = self.log_prob
            self.e_step(n, sample_array, sample_weights)
            self.m_step(n, sample_array)
            count += 1

    def print_params(self):
        for i in range(self.k):
            mean = self.means[i]
            cov = self.covariances[i]
            weight = self.weights[i]
            print('________________________________________\n')
            print('Component', i)
            print('Mean')
            print(mean)
            print('Covaraince')
            print(cov)
            print('Weight')
            print(weight, '\n')


class gmm:

    def __init__(self, k, tol=0.001, max_iters=1000):
        self.k = k
        self.tol = tol
        self.max_iters = max_iters
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.d = None
        self.p_nk = None
        self.log_prob = None
        self.N = 0

    def fit(self, sample_array, sample_weights=None, bounds=None, trunc_corr=False):
        if trunc_corr:
            sample_array, sample_weights = self.trunc_correction(sample_array, bounds, sample_weights)
        self.N, self.d = sample_array.shape
        model = estimator(self.k)
        model.fit(sample_array, sample_weights)
        self.means = model.means
        self.covariances = model.covariances
        self.weights = model.weights
        self.p_nk = model.p_nk
        self.log_prob = model.log_prob

    def match_components(self, new_model):
        orders = list(itertools.permutations(range(self.k), self.k))
        distances = np.empty(len(orders))
        index = 0
        for order in orders:
            dist = 0
            i = 0
            for j in order: # get sum of Euclidean distances between means
                #dist += np.sqrt(np.sum(np.square(self.means[i] - new_model.means[j])))
                # try Mahalanobis distance instead
                diff = new_model.means[j] - self.means[i]
                cov_inv = np.linalg.inv(self.covariances[i])
                temp_cov_inv = np.linalg.inv(new_model.covariances[j])
                dist += np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                dist += np.sqrt(np.dot(np.dot(diff, temp_cov_inv), diff))
                i += 1
            distances[index] = dist
            index += 1
        return orders[np.argmin(distances)] # returns order which gives minimum net Euclidean distance

    def merge(self, new_model, M):
        order = self.match_components(new_model)
        for i in range(self.k):
            j = order[i]
            old_mean = self.means[i]
            temp_mean = new_model.means[j]
            old_cov = self.covariances[i]
            temp_cov = new_model.covariances[j]
            old_weight = self.weights[i]
            temp_weight = new_model.weights[j]
            denominator = (self.N * old_weight) + (M * temp_weight) # this shows up a lot so just compute it once
            # start equation (6)
            mean = (self.N * old_weight * old_mean) + (M * temp_weight * temp_mean)
            mean /= denominator
            # start equation (7)
            cov1 = (self.N * old_weight * old_cov) + (M * temp_weight * temp_cov)
            cov1 /= denominator
            cov2 = (self.N * old_weight * old_mean * old_mean.T) + (M * temp_weight * temp_mean * temp_mean.T)
            cov2 /= denominator
            cov = cov1 + cov2 - mean * mean.T
            # start equation (8)
            weight = denominator / (self.N + M)
            # update everything
            self.means[i] = mean
            self.covariances[i] = cov
            self.weights[i] = weight

    def update(self, sample_array, sample_weights=None, bounds=None, trunc_corr=False):
        new_model = estimator(self.k, self.tol, self.max_iters)
        if trunc_corr:
            sample_array, sample_weights = self.trunc_correction(sample_array, bounds, sample_weights)
        new_model.fit(sample_array, sample_weights)
        M, _ = sample_array.shape
        self.merge(new_model, M)
        self.N += M

    def score(self, sample_array, bounds=None):
        n, _ = sample_array.shape
        scores = np.zeros((len(sample_array), 1))
        for i in range(self.k):
            w = self.weights[i]
            mean = self.means[i]
            cov = self.covariances[i]
            scores += np.rot90([multivariate_normal.pdf(x=sample_array, mean=mean, cov=cov)], -1) * w
        if bounds is not None:
            # we need to renormalize the pdf
            # to do this we sample from a full distribution (i.e. without truncation) and use the
            # fraction of samples that fall inside the bounds to renormalize
            full_sample_array = self.sample(n)
            llim = np.rot90(bounds[:,[0]])
            rlim = np.rot90(bounds[:,[1]])
            n1 = np.greater(full_sample_array, llim).all(axis=1)
            n2 = np.less(full_sample_array, rlim).all(axis=1)
            normalize = np.array(np.logical_and(n1, n2)).flatten()
            m = float(np.sum(normalize)) / n
            scores /= m
        return scores

    def sample(self, n, bounds=None):
        sample_array = np.empty((n, self.d))
        start = 0
        for component in range(self.k):
            w = self.weights[component]
            mean = self.means[component]
            cov = self.covariances[component]
            num_samples = int(n * w)
            if component == self.k -1:
                end = n
            else:
                end = start + num_samples
            if bounds is None:
                sample_array[start:end] = np.random.multivariate_normal(mean, cov, end - start)
            else:
                sample_array[start:end] = truncnorm.sample(mean, cov, bounds, end - start)
            start = end
        return sample_array

    def trunc_correction(self, sample_array, bounds, sample_weights=None):
        n, d = sample_array.shape
        if sample_weights is None:
            sample_weights = np.ones((n, 1))
        x_avg = [np.average(sample_array, axis=0, weights=sample_weights.flatten())]
        for dim in range(d):
            x_copy = np.copy(x_avg)
            llim = bounds[dim][0]
            rlim = bounds[dim][1]
            weight = np.sum(sample_weights) / (rlim - llim)
            ldiff = x_avg[0][dim] - llim
            rdiff = rlim - x_avg[0][dim]
            x_copy[0][dim] = llim - ldiff
            sample_array = np.append(sample_array, x_copy, axis=0)
            sample_weights = np.append(sample_weights, [[weight]], axis=0)
            x_copy[0][dim] = rlim + rdiff
            sample_array = np.append(sample_array, x_copy, axis=0)
            sample_weights = np.append(sample_weights, [[weight]], axis=0)
        return sample_array, sample_weights


    def print_params(self):
        for i in range(self.k):
            mean = self.means[i]
            cov = self.covariances[i]
            weight = self.weights[i]
            print('________________________________________\n')
            print('Component', i)
            print('Mean')
            print(mean)
            print('Covaraince')
            print(cov)
            print('Weight')
            print(weight, '\n')