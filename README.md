# Monte-Carlo-Integrator


Dependencies:

    python 2.7 or 3.6 +
    numpy
    scipy
    matplotlib

To test the monte carlo integrator alone, run

    python test.py

For a basic test of mcsampler, run

    python test_mcsampler_new.py

For a more thorough test, run

    python comprehensive_test.py [ndim] [ncomp] [model dimensions together (y/n)]

For example, to test in two dimensions with two Gaussian components, where dimensions are modeled together, run

    python comprehensive_test.py 2 2 y
