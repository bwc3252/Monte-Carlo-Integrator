from __future__ import print_function
import numpy as np
import gaussian_mixture_model as GMM


class integrator:
    '''
    Class to iteratively perform an adaptive Monte Carlo integral where the integrand
    is a combination of one or more Gaussian curves, in one or more dimensions.

    Parameters:

    d (int): Total number of dimensions.

    bounds (2 x d numpy array): Limits of integration, where each row represents
    [left_lim, right_lim] for its corresponding dimension.

    gmm_dict (dict): Dictionary where each key is a tuple of one or more dimensions
    that are to be modeled together. If the integrand has strong correlations between
    two or more dimensions, they should be grouped. Each value is by default initialized
    to None, and is replaced with the GMM object for its dimension(s).

    n_comp (int): The number of Gaussian components per group of dimensions.

    reflect (bool or 2 x d numpy array): Option to reflect all samples across limits of
    integration to help handle boundaries. If not False, must be a numpy array of the
    same shape as the bounds parameter, where each entry is a bool indicating whether
    or not to reflect over that boundary.

    trunc_corr (bool): An alternative option to correct for boundaries. Currently
    does not work.

    prior_samples (d x n numpy array): User-provided samples for intial evaluation
    and training. If None, a uniform prior is used. Note that if non-uniform prior
    is used, the prior_pdf must also be provided.

    prior_pdf (1 x n numpy array): User-provided responsibilities for prior samples
    '''

    def __init__(self, d, bounds, gmm_dict, n_comp, n=None, reflect=False, trunc_corr=False,
                    prior_samples=None, prior_pdf=None, user_func=None):
        # user-specified parameters
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.n_comp = n_comp
        self.reflect = reflect
        self.trunc_corr = trunc_corr
        self.prior_samples = prior_samples
        self.prior_pdf = prior_pdf
        self.user_func=user_func
        # constants
        self.t = 0.02 # percent estimated error threshold
        if n is None:
            self.n = (5000 * self.d) # number of samples per batch
        else:
            self.n = n
        # integrator object parameters
        self.sample_array = None
        self.value_array = None
        self.p_array = None
        self.integral = 0
        self.var = 0
        self.eff_samp = 0
        self.iterations = 0 # for weighted averages and count
        self.max_value = float('-inf') # for calculating eff_samp
        self.total_value = 0 # for calculating eff_samp

    def sample(self):
        if self.prior_samples is not None:
            self.sample_array, self.p_array = self.prior_samples, self.prior_pdf
            self.prior_samples, self.prior_pdf = None, None
            return
        self.p_array = np.ones((self.n, 1))
        self.sample_array = np.empty((self.n, self.d))
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = np.empty((len(dim_group), 2))
            index = 0
            for dim in dim_group:
                new_bounds[index] = self.bounds[dim]
                index += 1
            model = self.gmm_dict[dim_group]
            if model is None:
                # sample uniformly for this group of dimensions
                llim = new_bounds[:,0]
                rlim = new_bounds[:,1]
                temp_samples = np.random.uniform(llim, rlim, (self.n, len(dim_group)))
                # update responsibilities
                vol = np.prod(rlim - llim)
                self.p_array *= 1.0 / vol
            else:
                # sample from the gmm
                temp_samples = model.sample(self.n, new_bounds)
                # update responsibilities
                self.p_array *= model.score(temp_samples, new_bounds)
            index = 0
            for dim in dim_group:
                # put columns of temp_samples in final places in sample_array
                self.sample_array[:,[dim]] = temp_samples[:,[index]]
                index += 1

    def train(self):
        if self.reflect:
            # we need to reflect each dimension over the left and right limits
            sample_array, value_array, p_array, new_n_comp = self.reflect_over_bounds()
            new_n, _ = sample_array.shape
        else:
            sample_array, value_array, p_array = self.sample_array, self.value_array, self.p_array
            new_n_comp = self.n_comp
            new_n = self.n
        weights = abs(value_array / p_array) # training weights for samples
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = np.empty((len(dim_group), 2))
            index = 0
            for dim in dim_group:
                new_bounds[index] = self.bounds[dim]
                index += 1
            model = self.gmm_dict[dim_group] # get model for this set of dimensions
            temp_samples = np.empty((new_n, len(dim_group)))
            index = 0
            for dim in dim_group:
                # get samples corresponding to the current model
                temp_samples[:,[index]] = sample_array[:,[dim]]
                index += 1
            if model is None:
                # model doesn't exist yet
                model = GMM.gmm(new_n_comp)
                model.fit(temp_samples, sample_weights=weights, bounds=new_bounds)
            else:
                model.update(temp_samples, sample_weights=weights, bounds=new_bounds)
            self.gmm_dict[dim_group] = model


    def reflect_over_bounds(self):
        # make local copies of points, function values, and responsibilities
        sample_array = self.sample_array
        value_array = self.value_array
        p_array = self.p_array
        samples_to_append = np.empty((0, self.d))
        values_to_append = np.empty((0, 1))
        p_to_append = np.empty((0, 1))
        # get rotated arrays of function values and responsibilities
        rotated_values = np.flipud(value_array)
        rotated_p = np.flipud(p_array)
        new_n_comp = self.n_comp
        for dim in range(self.d): # each dimension will have two reflections
            if self.reflect[dim][1]: # right side needs reflected
                # get a copy of samples
                right = np.copy(sample_array)
                # reflect
                right[:,[dim]] *= -1
                # shift
                rlim = self.bounds[dim][0]
                right[:,[dim]] += 2 * rlim
                # append things
                samples_to_append = np.append(samples_to_append, right, axis=0)
                values_to_append = np.append(values_to_append, value_array, axis=0)
                p_to_append = np.append(p_to_append, p_array, axis=0)
                new_n_comp += self.n_comp
            if self.reflect[dim][0]: # left side needs reflected
                # get a copy of samples
                left = np.copy(sample_array)
                # reflect
                left[:,[dim]] *= -1
                # shift
                llim = self.bounds[dim][0]
                left[:,[dim]] += 2 * llim
                # append things
                samples_to_append = np.append(samples_to_append, left, axis=0)
                values_to_append = np.append(values_to_append, rotated_values, axis=0)
                p_to_append = np.append(p_to_append, rotated_p, axis=0)
                new_n_comp += self.n_comp
            # append things
            sample_array = np.append(sample_array, samples_to_append, axis=0)
            value_array = np.append(value_array, values_to_append, axis=0)
            p_array = np.append(p_array, p_to_append, axis=0)
        return sample_array, value_array, p_array, new_n_comp

    def calculate_results(self):
        # make local copies
        value_array = np.copy(self.value_array)
        p_array = np.copy(self.p_array)
        value_array /= p_array
        # get weight of current iteration
        weight = np.sum(value_array)
        # calculate variance
        curr_var = np.var(value_array) / self.n
        # calculate eff_samp
        max_value = np.max(value_array)
        if max_value > self.max_value:
            self.max_value = max_value
        self.total_value += np.sum(value_array)
        self.eff_samp = self.total_value / self.max_value
        # calculate integral
        curr_integral = (1.0 / self.n) * np.sum(value_array)
        # update results
        self.integral = ((self.integral * self.iterations) + curr_integral) / (self.iterations + 1)
        self.var = ((self.var * self.iterations) + curr_var) / (self.iterations + 1)


    def integrate(self, func, min_iter=10, max_iter=20, var_thresh=0.03, max_err=10):
        err_count = 0
        while self.iterations < max_iter:
            if err_count >= max_err:
                print('Exiting due to errors...')
                break
            try:
                self.sample()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                exit()
            except:
                print('Error sampling, retrying...')
                err_count += 1
                continue
            self.value_array = func(self.sample_array)
            self.calculate_results()
            #print(self.integral, '+/-', np.sqrt(self.var), 'with eff_samp', self.eff_samp)
            self.iterations += 1
            if self.iterations >= min_iter and self.var < var_thresh:
                break
            try:
                self.train()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                exit()
            except:
                print('Error training, retrying...')
                err_count += 1
            if self.user_func is not None:
                self.user_func(self)
