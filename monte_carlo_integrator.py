import numpy as np
import gaussian_mixture_model as GMM


class integrator:

    def __init__(self, d, bounds, gmm_dict, n_comp, reflect=False):
        # user-specified parameters
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.n_comp = n_comp
        self.reflect=reflect
        # constants
        self.t = 0.02 # percent estimated error threshold
        self.n = (5000 * self.d) # number of samples per batch
        # integrator object parameters
        self.sample_array = None
        self.value_array = None
        self.p_array = None
        self.integral = 0
        self.var = 0
        self.eff_samp = 0
        self.prev_weights_sum = 0 # for weighted averages

    def sample(self):
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
            sample_array, value_array, p_array = self.reflect_over_bounds()
            new_n_comp = self.n_comp * (3 ** self.d) # reflection increases number of components
            new_n, _ = sample_array.shape
        else:
            sample_array, value_array, p_array = self.sample_array, self.value_array, self.p_array
            new_n_comp = self.n_comp
            new_n = self.n
        weights = value_array / p_array # training weights for samples
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
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
                model.fit(temp_samples, sample_weights=weights)
            else:
                model.update(temp_samples, sample_weights=weights)
            self.gmm_dict[dim_group] = model
            #model.print_params()

    def reflect_over_bounds(self):
        # make local copies of points, function values, and responsibilities
        sample_array = self.sample_array
        value_array = self.value_array
        p_array = self.p_array
        # get rotated arrays of function values and responsibilities
        rotated_values = np.flipud(value_array)
        rotated_p = np.flipud(p_array)
        for dim in range(self.d): # each dimension will have two reflections
            # get reflections of points in this dimension
            right = sample_array
            right[:,[dim]] *= -1
            left = right
            # get limits to reflect over
            llim = self.bounds[dim][0]
            rlim = self.bounds[dim][1]
            # x_right = (rlim - x) + rlim = (-1 * x) + (2 * rlim)
            # x_left = llim - (x - llim) = (-1 * x) + (2 * llim)
            right[:,[dim]] += 2 * rlim
            left[:,[dim]] += 2 * llim
            # append new values to all arrays
            sample_array = np.append(sample_array, right, axis=0)
            sample_array = np.append(sample_array, left, axis=0)
            value_array = np.append(value_array, value_array, axis=0)
            value_array = np.append(value_array, rotated_values, axis=0)
            p_array = np.append(p_array, p_array, axis=0)
            p_array = np.append(p_array, rotated_p, axis=0)
        return sample_array, value_array, p_array

    def calculate_results(self):
        # make local copies
        value_array = self.value_array
        p_array = self.p_array
        # get weight of current iteration
        weight = np.sum(p_array)
        # calculate variance
        curr_var = np.var(value_array)
        # calculate eff_samp
        curr_eff_samp = np.sum(value_array) / np.max(value_array)
        # calculate integral
        value_array /= p_array
        curr_integral = (1.0 / self.n) * np.sum(value_array)
        # update results
        self.integral = ((self.integral * self.prev_weights_sum) + (curr_integral * weight)) / (self.prev_weights_sum + weight)
        self.var = ((self.var * self.prev_weights_sum) + (curr_var * weight)) / (self.prev_weights_sum + weight)
        self.eff_samp = ((self.eff_samp * self.prev_weights_sum) + (curr_eff_samp * weight)) / (self.prev_weights_sum + weight)
        # update weight sum
        self.prev_weights_sum += weight


    def integrate(self, func, min_iter=10, max_iter=20, var_thresh=0.03):
        count = 0
        while count < max_iter:
            self.sample()
            self.value_array = func(self.sample_array)
            self.train()
            self.calculate_results()
            print(self.integral, '+/-', np.sqrt(self.var), 'with eff_samp', self.eff_samp)
            count += 1
            if count >= min_iter and self.var < var_thresh:
                break
