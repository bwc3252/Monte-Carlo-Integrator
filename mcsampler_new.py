from __future__ import print_function
import sys
import math
import bisect
from collections import defaultdict

import numpy as np

import itertools
import functools

#from statutils import cumvar

import monte_carlo_integrator as monte_carlo

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>, Ben Champion <bwc3252@rit.edu>"

rosDebugMessages = True

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCSampler(object):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """

    @staticmethod
    def match_params_from_args(args, params):
        """
        Given two unordered sets of parameters, one a set of all "basic" elements
        (strings) possible, and one a set of elements both "basic" strings and
        "combined" (basic strings in tuples), determine whether the sets are equivalent
        if no basic element is repeated.

        e.g. set A ?= set B

        ("a", "b", "c") ?= ("a", "b", "c") ==> True
        (("a", "b", "c")) ?= ("a", "b", "c") ==> True
        (("a", "b"), "d")) ?= ("a", "b", "c") ==> False  # basic element 'd' not in set B
        (("a", "b"), "d")) ?= ("a", "b", "d", "c") ==> False  # not all elements in set B
        represented in set A
        """
        not_common = set(args) ^ set(params)
        if len(not_common) == 0:
            # All params match
            return True
        if all([not isinstance(i, tuple) for i in not_common]):
            # The only way this is possible is if there are
            # no extraneous params in args
            return False

        to_match = filter(lambda i: not isinstance(i, tuple), not_common)
        against = filter(lambda i: isinstance(i, tuple), not_common)

        matched = []
        import itertools
        for i in range(2, max(map(len, against))+1):
            matched.extend([t for t in itertools.permutations(to_match, i) if t in against])
        return (set(matched) ^ set(against)) == set()


    def __init__(self):
        # Total number of samples drawn
        self.ntotal = 0
        # Samples per iteration
        self.n = 0
        # Parameter names
        self.params = set()
        self.params_ordered = []  # keep them in order. Important to break likelihood function need for names
        # parameter -> pdf function object
        self.pdf = {}
        # If the pdfs aren't normalized, this will hold the normalization
        # constant
        self._pdf_norm = defaultdict(lambda: 1)
        # Cache for the sampling points
        self._rvs = {}
        # parameter -> cdf^{-1} function object
        self.cdf = {}
        self.cdf_inv = {}
        # params for left and right limits
        self.llim, self.rlim = {}, {}
        # Keep track of the adaptive parameters
        self.adaptive = []

        # Keep track of the adaptive parameter 1-D marginalizations
        self._hist = {}

        # MEASURES (=priors): ROS needs these at the sampler level, to clearly separate their effects
        # ASSUMES the user insures they are normalized
        self.prior_pdf = {}

        self.func = None
        self.sample_format = None
        self.curr_args = None


    def clear(self):
        """
        Clear out the parameters and their settings, as well as clear the sample cache.
        """
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1.0)
        self._rvs = {}
        self._hist = {}
        self.cdf = {}
        self.cdf_inv = {}
        self.llim = {}
        self.rlim = {}
        self.adaptive = []

    def add_parameter(self, params, pdf=None,  cdf_inv=None, left_limit=None, right_limit=None,
                        prior_pdf=None, adaptive_sampling=False):
        """
        Add one (or more) parameters to sample dimensions. params is either a string
        describing the parameter, or a tuple of strings. The tuple will indicate to
        the sampler that these parameters must be sampled together. left_limit and
        right_limit are on the infinite interval by default, but can and probably should
        be specified. If several params are given, left_limit, and right_limit must be a
        set of tuples with corresponding length. Sampling PDF is required, and if not
        provided, the cdf inverse function will be determined numerically from the
        sampling PDF.
        """
        self.params.add(params) # does NOT preserve order in which parameters are provided
        self.params_ordered.append(params)
        if rosDebugMessages:
            print(" mcsampler: Adding parameter ", params, " with limits ", [left_limit, right_limit])
        if isinstance(params, tuple):
            assert all(map(lambda lim: lim[0] < lim[1], zip(left_limit, right_limit)))
            if left_limit is None:
                self.llim[params] = list(float("-inf"))*len(params)
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = list(float("+inf"))*len(params)
            else:
                self.rlim[params] = right_limit
        else:
            assert left_limit < right_limit
            if left_limit is None:
                self.llim[params] = float("-inf")
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = float("+inf")
            else:
                self.rlim[params] = right_limit

    def evaluate(self, samples):
        '''
        Interfaces between monte_carlo_integrator sample format (1 (n x d) array)
        and likelihood function sample format (d 1D arrays in a list)
        '''
        # integrand expects a list of 1D rows
        temp = []
        for index in range(len(self.curr_args)):
            temp.append(samples[:,index])
        temp_ret = self.func(*temp)
        return np.rot90([temp_ret], -1) # monte_carlo_integrator expects a column


    def calc_pdf(self, samples):
        '''
        Similar to evaluate(), interfaces between sample formats. Must also handle
        possibility of no prior for one of more dimensions
        '''
        n, _ = samples.shape
        temp_ret = np.ones((n, 1))
        # pdf functions expect 1D rows
        for index in range(len(self.curr_args)):
            if self.curr_args[index] in self.pdf:
                pdf_func = self.pdf[self.curr_args[index]]
                temp_samples = samples[:,index]
                # monte carlo integrator expects a column
                temp_ret *= np.rot90([pdf_func(temp_samples)], -1)
        return temp_ret


    def integrate(self, func, args, direct_eval=True, n_comp=None, n=None, nmax=None, write_to_file=False,
                gmm_dict=None, var_thresh=0.05, min_iter=10, max_iter=20, neff=float('inf'), reflect=False,
                mcsamp_func=None, integrator_func=None, proc_count=None):
        '''
        Integrate the specified function over the specified parameters.

        func: function to integrate

        args: list of parameters to integrate over

        direct_eval (bool): whether func can be evaluated directly with monte_carlo_integrator
        format or not

        n_comp: number of gaussian components for model

        n: number of samples per iteration

        nmax: maximum number of samples for all iterations

        write_to_file (bool): write data to file

        gmm_dict: dictionary of dimensions and mixture models (see monte_carlo_integrator
        documentation for more)

        var_thresh: result variance threshold for termination

        min_iter: minimum number of integrator iterations

        max_iter: maximum number of integrator iterations

        neff: eff_samp cutoff for termination

        reflect (bool): whether or not to reflect samples over boundaries (you should
        basically never use this, it's really slow)

        mcsamp_func: function to be executed before mcsampler_new terminates (for example,
        to print results or debugging info)

        integrator_func: function to be executed each iteration of the integrator (for
        example, to print intermediate results)

        proc_count: size of multiprocessing pool. set to None to not use multiprocessing
        '''

        # set up a lot of preliminary stuff
        self.func = func
        self.curr_args = args
        if n_comp is None:
            print('No n_comp given, assuming 1 component per dimension')
            n_comp = 1
        dim = len(args)
        bounds = []
        for param in args:
            bounds.append([self.llim[param], self.rlim[param]])
        bounds = np.array(bounds)

        # for now, we hardcode the assumption that there are no correlated dimensions,
        # unless otherwise specified

        if gmm_dict is None:
            gmm_dict = {}
            for x in range(dim):
                gmm_dict[(x,)] = None

        # do the integral

        integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, n=n, prior=self.calc_pdf,
                        reflect=reflect, user_func=integrator_func, proc_count=proc_count)
        if not direct_eval:
            func = self.evaluate
        integrator.integrate(func, min_iter=min_iter, max_iter=max_iter, var_thresh=var_thresh, neff=neff, nmax=nmax)

        # get results

        self.n = integrator.n
        self.ntotal = integrator.ntotal
        integral = integrator.integral
        var = integrator.var
        eff_samp = integrator.eff_samp
        sample_array = integrator.sample_array
        value_array = integrator.value_array
        p_array = integrator.p_array
        prior_array = integrator.prior_array

        # user-defined function
        if mcsamp_func is not None:
            mcsamp_func(self, integrator)

        # populate dictionary

        index = 0
        for param in args:
            self._rvs[param] = sample_array[:,[index]]
            index += 1
        self._rvs['joint_prior'] = prior_array
        self._rvs['joint_s_prior'] = p_array
        self._rvs['integrand'] = value_array

        # write data to file

        if write_to_file:
            dat_out = np.c_[sample_array, value_array, p_array]
            np.savetxt('mcsampler_data.txt', dat_out,
                        header=" ".join(['sample_array', 'value_array', 'p_array']))

        return integral, var, eff_samp, {}


def inv_uniform_cdf(a, b, x):
    return (b-a)*x+a

def gauss_samp(mu, std, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2) + myfloor

#gauss_samp_withfloor_vector = np.vectorize(gauss_samp_withfloor,excluded=['mu','std','myfloor'],otypes=[np.float])
gauss_samp_withfloor_vector = np.vectorize(gauss_samp_withfloor,otypes=[np.float])


# Mass ratio. PDF propto 1/(1+q)^2.  Defined so mass ratio is < 1
# expr = Integrate[1/(1 + q)^2, q]
# scale = (expr /. q -> qmax )  - (expr /. q -> qmin)
# (expr - (expr /. q -> qmin))/scale == x // Simplify
# q /. Solve[%, q][[1]] // Simplify
# % // CForm
def q_samp_vector(qmin,qmax,x):
    scale = 1./(1+qmin) - 1./(1+qmax)
    return 1/np.power((1+x),2)/scale
def q_cdf_inv_vector(qmin,qmax,x):
    return np.array((qmin + qmax*qmin + qmax*x - qmin*x)/(1 + qmax - qmax*x + qmin*x),dtype=np.float128)

# total mass. Assumed used with q.  2M/Mmax^2-Mmin^2
def M_samp_vector(Mmin,Mmax,x):
    scale = 2./(Mmax**2 - Mmin**2)
    return x*scale


def cos_samp(x):
        return np.sin(x)/2   # x from 0, pi

def dec_samp(x):
        return np.sin(x+np.pi/2)/2   # x from 0, pi

cos_samp_vector = np.vectorize(cos_samp,otypes=[np.float])
dec_samp_vector = np.vectorize(dec_samp,otypes=[np.float])
def cos_samp_cdf_inv_vector(p):
    return np.arccos( 2*p-1)   # returns from 0 to pi
def dec_samp_cdf_inv_vector(p):
    return np.arccos(2*p-1) - np.pi/2  # target from -pi/2 to pi/2


def pseudo_dist_samp(r0,r):
        return r*r*np.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01  # put a floor on probability, so we converge. Note this floor only cuts out NEARBY distances

#pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[np.float])
pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float])

def delta_func_pdf(x_0, x):
    return 1.0 if x == x_0 else 0.0

delta_func_pdf_vector = np.vectorize(delta_func_pdf, otypes=[np.float])

def delta_func_samp(x_0, x):
    return x_0

delta_func_samp_vector = np.vectorize(delta_func_samp, otypes=[np.float])

class HealPixSampler(object):
    """
    Class to sample the sky using a FITS healpix map. Equivalent to a joint 2-D pdf in RA and dec.
    """

    @staticmethod
    def thph2decra(th, ph):
        """
        theta/phi to RA/dec
        theta (north to south) (0, pi)
        phi (east to west) (0, 2*pi)
        declination: north pole = pi/2, south pole = -pi/2
        right ascension: (0, 2*pi)

        dec = pi/2 - theta
        ra = phi
        """
        return np.pi/2-th, ph

    @staticmethod
    def decra2thph(dec, ra):
        """
        theta/phi to RA/dec
        theta (north to south) (0, pi)
        phi (east to west) (0, 2*pi)
        declination: north pole = pi/2, south pole = -pi/2
        right ascension: (0, 2*pi)

        theta = pi/2 - dec
        ra = phi
        """
        return np.pi/2-dec, ra

    def __init__(self, skymap, massp=1.0):
        self.skymap = skymap
        self._massp = massp
        self.renormalize()

    @property
    def massp(self):
        return self._massp

    @massp.setter
    def massp(self, value):
        assert 0 <= value <= 1
        self._massp = value
        norm = self.renormalize()

    def renormalize(self):
        """
        Identify the points contributing to the overall cumulative probability distribution, and set the proper normalization.
        """
        res = healpy.npix2nside(len(self.skymap))
        self.pdf_sorted = sorted([(p, i) for i, p in enumerate(self.skymap)], reverse=True)
        self.valid_points_decra = []
        cdf, np = 0, 0
        for p, i in self.pdf_sorted:
            if p == 0:
                continue # Can't have a zero prior
            self.valid_points_decra.append(HealPixSampler.thph2decra(*healpy.pix2ang(res, i)))
            cdf += p
            if cdf > self._massp:
                break
        self._renorm = cdf
        # reset to indicate we'd need to recalculate this
        self.valid_points_hist = None
        return self._renorm

    def __expand_valid(self, min_p=1e-7):
        #
        # Determine what the 'quanta' of probabilty is
        #
        if self._massp == 1.0:
            # This is to ensure we don't blow away everything because the map
            # is very spread out
            min_p = min(min_p, max(self.skymap))
        else:
            # NOTE: Only valid if CDF descending order is kept
            min_p = self.pseudo_pdf(*self.valid_points_decra[-1])

        self.valid_points_hist = []
        ns = healpy.npix2nside(len(self.skymap))

        # Renormalize first so that the vector histogram is properly normalized
        self._renorm = 0
        # Account for probability lost due to cut off
        for i, v in enumerate(self.skymap >= min_p):
            self._renorm += self.skymap[i] if v else 0

        for pt in self.valid_points_decra:
            th, ph = HealPixSampler.decra2thph(pt[0], pt[1])
            pix = healpy.ang2pix(ns, th, ph)
            if self.skymap[pix] < min_p:
                continue
            self.valid_points_hist.extend([pt]*int(round(self.pseudo_pdf(*pt)/min_p)))
        self.valid_points_hist = np.array(self.valid_points_hist).T

    def pseudo_pdf(self, dec_in, ra_in):
        """
        Return pixel probability for a given dec_in and ra_in. Note, uses healpy functions to identify correct pixel.
        """
        th, ph = HealPixSampler.decra2thph(dec_in, ra_in)
        res = healpy.npix2nside(len(self.skymap))
        return self.skymap[healpy.ang2pix(res, th, ph)]/self._renorm

    def pseudo_cdf_inverse(self, dec_in=None, ra_in=None, ndraws=1, stype='vecthist'):
        """
        Select points from the skymap with a distribution following its corresponding pixel probability. If dec_in, ra_in are suupplied, they are ignored except that their shape is reproduced. If ndraws is supplied, that will set the shape. Will return a 2xN np array of the (dec, ra) values.
        stype controls the type of sampling done to retrieve points. Valid choices are
        'rejsamp': Rejection sampling: accurate but slow
        'vecthist': Expands a set of points into a larger vector with the multiplicity of the points in the vector corresponding roughly to the probability of drawing that point. Because this is not an exact representation of the proability, some points may not be represented at all (less than quantum of minimum probability) or inaccurately (a significant fraction of the fundamental quantum).
        """

        if ra_in is not None:
            ndraws = len(ra_in)
        if ra_in is None:
            ra_in, dec_in = np.zeros((2, ndraws))

        if stype == 'rejsamp':
            # FIXME: This is only valid under descending ordered CDF summation
            ceiling = max(self.skymap)
            i, np = 0, len(self.valid_points_decra)
            while i < len(ra_in):
                rnd_n = np.random.randint(0, np)
                trial = np.random.uniform(0, ceiling)
                if trial <= self.pseudo_pdf(*self.valid_points_decra[rnd_n]):
                    dec_in[i], ra_in[i] = self.valid_points_decra[rnd_n]
                    i += 1
            return np.array([dec_in, ra_in])
        elif stype == 'vecthist':
            if self.valid_points_hist is None:
                self.__expand_valid()
            np = self.valid_points_hist.shape[1]
            rnd_n = np.random.randint(0, np, len(ra_in))
            dec_in, ra_in = self.valid_points_hist[:,rnd_n]
            return np.array([dec_in, ra_in])
        else:
            raise ValueError("%s is not a recgonized sampling type" % stype)

#pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[np.float])
pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float])


def sanityCheckSamplerIntegrateUnity(sampler,*args,**kwargs):
        return sampler.integrate(lambda *args: 1,*args,**kwargs)

###
### CONVERGENCE TESTS
###


# neff by another name:
#    - value: tests for 'smooth' 1-d cumulative distributions
#    - require  require the most significant-weighted point be less than p of all cumulative probability
#    - this test is *equivalent* to neff > 1/p
#    - provided to illustrate the interface
def convergence_test_MostSignificantPoint(pcut, rvs, params):
    weights = rvs["weights"] #rvs["integrand"]* rvs["joint_prior"]/rvs["joint_s_prior"]
    indxmax = np.argmax(weights)
    wtSum = np.sum(weights)
    return  weights[indxmax]/wtSum < pcut


# normality test: is the MC integral normally distributed, with a small standard deviation?
#    - value: tests for converged integral
#    - arguments:
#         - ncopies:               # of sub-integrals
#         - pcutNormalTest     Threshold p-value for normality test
#         - sigmaCutErrorThreshold   Threshold relative error in the integral
#    - implement normality test on **log(integral)** since the log should also be normally distributed if well converged
#           - this helps us handle large orders-of-magnitude differences
#           - compatible with a *relative* error threshold on integral
#           - only works for *positive-definite* integrands
#    - other python normality tests:
#          scipy.stats.shapiro
#          scipy.stats.anderson
#  WARNING:
#    - this test assumes *unsorted* past history: the 'ncopies' segments are assumed independent.
import scipy.stats as stats
def convergence_test_NormalSubIntegrals(ncopies, pcutNormalTest, sigmaCutRelativeErrorThreshold, rvs, params):
    weights = rvs["integrand"]* rvs["joint_prior"]/rvs["joint_s_prior"]  # rvs["weights"] # rvs["weights"] is *sorted* (side effect?), breaking test. Recalculated weights are not.  Use explicitly calculated weights until sorting effect identified
#    weights = weights /np.sum(weights)    # Keep original normalization, so the integral values printed to stdout have meaning relative to the overall integral value.  No change in code logic : this factor scales out (from the log, below)
    igrandValues = np.zeros(ncopies)
    len_part = np.int(len(weights)/ncopies)  # deprecated: np.floor->np.int
    for indx in np.arange(ncopies):
        igrandValues[indx] = np.log(np.mean(weights[indx*len_part:(indx+1)*len_part]))  # change to mean rather than sum, so sub-integrals have meaning
    igrandValues= np.sort(igrandValues)#[2:]                            # Sort.  Useful in reports
    valTest = stats.normaltest(igrandValues)[1]                              # small value is implausible
    igrandSigma = (np.std(igrandValues))/np.sqrt(ncopies)   # variance in *overall* integral, estimated from variance of sub-integrals
    print(" Test values on distribution of log evidence:  (gaussianity p-value; standard deviation of ln evidence) ", valTest, igrandSigma)
    print(" Ln(evidence) sub-integral values, as used in tests  : ", igrandValues)
    return valTest> pcutNormalTest and igrandSigma < sigmaCutRelativeErrorThreshold   # Test on left returns a small value if implausible. Hence pcut ->0 becomes increasingly difficult (and requires statistical accidents). Test on right requires relative error in integral also to be small when pcut is small.   FIXME: Give these variables two different names
