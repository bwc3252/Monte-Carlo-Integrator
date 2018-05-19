# Monte-Carlo-Integrator


Numpy and scikit-learn need to be installed, and the following files must be in
the working directory:

    monte_carlo_integrator.py
    weighted_gmm.py
    mcsamper_new.py

To test the monte carlo integrator alone, you need

    test.py

To test mcsampler, you need

    test_mcsampler_new.py

To test mcsampler and generate a CDF of the sampled points you need

    test_distribution.py

Each test file should run on its own. Run:
    
    python3 <test_file.py>

with the appropriate test file.

The integrand, the number of dimensions, the limits of integration, and the grouping of dimensions can all be changed in the test files.
