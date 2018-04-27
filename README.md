# Monte-Carlo-Integrator


Numpy and scikit-learn need to be installed, and the following files must be in
the working directory:

    monte_carlo_integrator.py
    weighted_gmm.py

test.py must also be in the working directory if its functionality is desired.


To test the integrator, run

    python3 test.py

The default test integral should evaluate to 2.  The integrand, the number of
dimensions, the limits of integration, and the grouping of dimensions can all
be changed in test.py.
