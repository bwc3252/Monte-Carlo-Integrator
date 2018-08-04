import numpy as np
from time import time
import warnings
import dill
from scipy.stats import multivariate_normal

#import weighted_gmm # a modified implementation of the scikit-learn gmm
import multivariate_truncnorm as truncnorm # to sample from a truncated multivariate normal distribution
import gaussian_mixture_model as GMM # a custom gmm implementation that supports online updating

class integrator:
    '''

    A class for evaluating multivariate integrals using an adaptive monte carlo method.
    It is assumed that the integrand can be well-approximated by a mixture of Gaussians.

    Parameters

    d: Number of dimensions

    bounds: A (d x 2) array where each row is [lower_bound, upper_bound] for its
    corresponding dimension

    gmm_dict: dictionary where each key is a tuple of dimension indicies and each value
    is a either a mixture model object or None.  Must contain all dimensions with no repeats.

    n_comp: number of gaussian components for each dimension

    After creating an instance of this object and defining func to be a vectorized python
    function, calling monte_carlo_integrator(func) will return the integral.

    '''

    def __init__(self, d, bounds, gmm_dict, n_comp):
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.n_comp = n_comp
        self.t = 0.02 # percent estimated error threshold
        self.n = (5000 * self.d) # this is a VERY arbitrary choice about the number of
                                 # samples that are necessary

    def uniform(self, func):
        '''
        Uniformly samples the function, returning n d-dimensional samples and the function
        value at those samples.
        '''
        n = self.n
        d = self.d
        bounds = self.bounds
        llim_array = np.rot90(bounds[:,[0]])
        rlim_array = np.rot90(bounds[:,[1]])
        sample_array = np.random.uniform(llim_array, rlim_array, (n, d))
        value_array = func(sample_array)
        return sample_array, value_array

    def sample_from_gmm(self, gmm_dict, func):
        '''
        Samples each dimension according to the Gaussian Mixture Model for that dimension.
        If no mixture model exists, samples uniformly. Returns the n x d array of samples,
        the responsibility of each sample according to the model, and the function value
        for each sample.
        '''
        n = self.n
        d = self.d
        bounds = self.bounds
        llim_array = np.rot90(bounds[:,[0]])
        sample_array = np.empty((n, d))
        sample_array_i = np.empty((n, d))
        p_array = np.ones((n, 1))
        p_array_i = np.ones((n, 1))
        for dim_group in gmm_dict:
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = np.empty((len(dim_group), 2))
            index = 0
            for dim in dim_group:
                new_bounds[index] = bounds[dim]
                index += 1
            clf = gmm_dict[dim_group]
            # create empty intermediate arrays for samples
            sample_column = np.empty((n, len(dim_group)))
            sample_column_i = np.empty((n, len(dim_group)))
            if clf is not None:
                means, covariances, weights = clf.means, clf.covariances, clf.weights
                start = 0
                end = 0
                for component in range(len(means)):
                    mean = means[component]
                    cov = covariances[component]
                    weight = weights[component]
                    interval = int(weight * n)
                    if component == len(means) - 1:
                        end = n
                    else:
                        end += interval
                    sample_column[start:end] = np.random.multivariate_normal(mean, cov, end - start)
                    sample_column_i[start:end] = truncnorm.sample(mean, cov, new_bounds, end - start)
                    start = end + 1
                p_array *= clf.score(sample_array) # np.rot90([np.exp(clf.score(sample_column))], -1)
                p_array_i *= clf.score(sample_array_i) # np.rot90([np.exp(clf.score(sample_column_i))], -1)
            else:
                llim = np.rot90(new_bounds[:,[0]])
                rlim = np.rot90(new_bounds[:,[1]])
                sample_column = np.random.uniform(llim, rlim, (n, len(dim_group)))
                sample_column_i = sample_column
                vol = np.prod(rlim - llim)
                p_array *= 1.0 / vol
            index = 0
            for dim in dim_group:
                sample_array[:,[dim]] = sample_column[:,[index]]
                sample_array_i[:,[dim]] = sample_column_i[:,[index]]
                index += 1
        value_array = func(sample_array)
        value_array_i = func(sample_array_i)
        return sample_array, sample_array_i, p_array, p_array_i, value_array, value_array_i

    def calc_integral(self, sample_array, value_array, p_array, sample_array_i, value_array_i, p_array_i):
        '''
        Performs the monte carlo integration for the given function values and responsibilities.
        '''
        d = self.d
        bounds = self.bounds
        n = self.n
        # normalize n to account for truncation
        llim = np.rot90(bounds[:,[0]])
        rlim = np.rot90(bounds[:,[1]])
        n1 = np.greater(sample_array, llim).all(axis=1)
        n2 = np.less(sample_array, rlim).all(axis=1)
        normalize = np.array(np.logical_and(n1, n2)).flatten()
        k = float(np.sum(normalize)) / n
        # do integration
        value_array_i /= p_array_i
        i = np.sum(value_array_i)
        return (k / n) * i

    def fit_gmm(self, sample_array, value_array, gmm_dict, p_array):
        '''
        Attempts to fit a Gaussian Mixture Model to the data.
        '''
        n = self.n
        d = self.d
        n_comp = self.n_comp
        #t = time()
        weights = abs(value_array / p_array)
        for dim_group in gmm_dict:
            exists = True
            clf = gmm_dict[dim_group]
            if not clf:
                exists = False
                clf = GMM.gmm(n_comp) # weighted_gmm.WeightedGMM(n_components=n_comp, covariance_type='full')
            samples_to_fit = np.empty((n, len(dim_group)))
            index = 0
            for dim in dim_group:
                samples_to_fit[:,[index]] = sample_array[:,[dim]]
                index += 1
            #try:
            if exists:
                clf.update(sample_array=samples_to_fit, sample_weights=weights)
            else:
                clf.fit(sample_array=samples_to_fit, sample_weights=weights)
            gmm_dict[dim_group] = clf
            '''
            except KeyboardInterrupt:
                return False
            except:
                # mixture model failed to fit, revert to uniform sampling
                print('Failed to fit GMM')
                gmm_dict[dim_group] = None
            '''
            if clf:
                if not True: #clf.converged_:
                    print('Failed to fit GMM')
                    # mixture model failed to fit, revert to uniform sampling
                    gmm_dict[dim_group] = None
        #print('time to train:', round(time() - t, 2), 'seconds')
        return gmm_dict

    def calculate_error(self, sample_array, value_array):
        '''
        I'm not really sure if this is correct but it returns results that seem
        reasonable.
        '''
        n = self.n
        d = self.d

        x = np.average(np.square(value_array))
        y = np.square(np.average(value_array))
        err_squared = (x - y) / n
        return err_squared

    def integrate(self, func, err_thresh=None, max_count=15):
        '''
        Main function to sample the integrand, model each dimension's distribution, and
        iteratively integrate the function and re-train the model until convergence is reached.

        func: integrand function (must be able to take numpy array as parameter)

        returns: integral, error, list of sample arrays
        '''
        n = self.n
        d = self.d
        t = self.t
        gmm_dict = self.gmm_dict
        sample_array, value_array = self.uniform(func)
        count = 0
        total_integral = 0
        target_count = 10
        total_iters = 0
        integral_list = np.array([])
        weight_list = np.array([])
        error_list = np.array([])
        eff_samp_list = np.array([])
        sample_array_list = []
        p_array_list = []
        value_array_list = []
        p_array = np.ones((n, 1))
        while True:
            total_iters += 1

            # the version of the scikit-learn mixture model in use here calls a deprecated
            # function in scikit-learn, which causes it to print a bunch of annoying warnings.
            # we don't need this for now, so we ignore the warnings.

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # fit the model
                gmm_dict = self.fit_gmm(sample_array, value_array, gmm_dict, p_array)
                if not gmm_dict:
                    break
                # sample from newly-fitted GMM
                sample_array, sample_array_i, p_array, p_array_i, value_array, value_array_i = self.sample_from_gmm(gmm_dict, func)
            integral = self.calc_integral(sample_array, value_array, p_array, sample_array_i, value_array_i, p_array_i)
            print(integral)
            print()
            err_squared = self.calculate_error(sample_array_i, value_array_i)
            err = np.sqrt(err_squared)
            # (rough) method to check for convergence
            if (err / integral) < t:
                eff_samp = np.sum(value_array_i) / np.max(value_array_i)
                integral_list = np.append(integral_list, integral)
                error_list = np.append(error_list, err_squared)
                weight_list = np.append(weight_list, np.sum(p_array_i))
                eff_samp_list = np.append(eff_samp_list, eff_samp)
                value_array_list.append(value_array_i)
                sample_array_list.append(sample_array_i)
                p_array_list.append(p_array)
                temp_weight_list = weight_list / np.sum(weight_list)
                temp_error_list = error_list * temp_weight_list
                running_error = np.sqrt(np.sum(temp_error_list))
                count += 1
                print('error:', err)
                print()
            if count >= target_count:
                if err_thresh is not None:
                    if running_error < err_thresh:
                        break
                else:
                    break
            if count > max_count:
                print('max_count reached before error threshold')
                break
        weight_list /= np.sum(weight_list)
        integral_list *= weight_list
        error_list *= weight_list
        eff_samp_list *= weight_list
        return {'integral':np.sum(integral_list), 'error':running_error,
                'sample_array':sample_array_list, 'value_array':value_array_list,
                'p_array':p_array_list, 'eff_samp':np.sum(eff_samp_list)}
