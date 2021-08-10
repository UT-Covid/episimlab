#!python
#cython: boundscheck=True
#cython: cdivision=True
#cython: infertypes=False
#cython: initializedcheck=False
#cython: nonecheck=True
#cython: wraparound=False
#distutils: language = c
#distutils: extra_link_args = ['-lgsl', '-lgslcblas', '-fopenmp']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration -Wno-nonnull -Wno-nullability-completeness

import logging
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# Abbreviate numpy dtypes
DTYPE_FLOAT = np.float64
DTYPE_INT = np.intc

cdef gsl_rng *get_seeded_rng(int int_seed) nogil:
    """Returns a C pointer to instance of MT-19937 generator
    (https://www.gnu.org/software/gsl/doc/html/rng.html#c.gsl_rng_mt19937).
    Seeds with int32 seed `int_seed`.
    """
    cdef:
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, int_seed)
    return rng

# binomial distribution from GSL lib
#cdef extern from "gsl/gsl_randist.h" nogil:
#    unsigned int gsl_ran_binomial(gsl_rng * r, double p, unsigned int n)

# logarithm (base e)
#cdef extern from "math.h":
#    double complex log(double complex x) nogil

def discrete_time_approx_wrapper(float rate, float timestep):
    """Thin Python wrapper around the below C function
    """
    return discrete_time_approx(rate, timestep)


def test_gsl_poisson(double mu, int int_seed, int iters, int enable_omp):
    """For testing purposes only
    TODO
    """
    cdef:
        gsl_rng *rng = get_seeded_rng(int_seed)
        Py_ssize_t iter_len = iters
        Py_ssize_t i
        np.ndarray result_arr = np.zeros((iters), dtype=DTYPE_INT)
        int [:] rv = result_arr

    if enable_omp == 0:
        for i in range(iters):
            rv[i] = gsl_ran_poisson(rng, mu)
    else:
        for i in prange(iters, nogil=True):
            rv[i] = gsl_ran_poisson(rng, mu)
    return result_arr

cdef double discrete_time_approx(double rate, double timestep) nogil:
        """
        :param rate: daily rate
        :param timestep: timesteps per day
        :return: rate rescaled by time step
        """
        # if rate >= 1:
            # return np.nan
        # elif timestep == 0:
            # return np.nan

        return 1. - (1. - rate)**(1. / timestep)

# # define "e" from complex numbers; see https://stackoverflow.com/questions/37757067/using-c-complex-functions-in-cython
# cdef extern from "<complex.h>" namespace "std":
#     double complex exp(double complex z)
#     float complex exp(float complex z)  # overload
#
# # aka inverse logit
# cdef double complex expit(double rate):
#
#     return exp(rate)/(exp(rate) + 1.0)
#
# cdef double complex sigmoid(double x):
#
#     return (2.0*expit(2.0*(x))-1.0)
#
# cdef double complex discrete_time_sigmoid_approx(double rate_param, double timestep):
#
#     # convert rates to probabilities and discretize for binomial transitions
#     dt = 1.0/timestep
#
#     return 1.0 - exp(log(1.0-(sigmoid(rate_param))) * dt)
#
# # TO DO: refactor
# cdef double BIN_dt(unsigned int N, float discrete_rate, gsl_rng *rng):
#
#     if N > 0.5:
#         count = gsl_ran_binomial(rng, discrete_rate, N)
#     else:
#         count = 0.0
#     return count
#
# cdef double calc_binom_pop_change(unsigned int N, float discrete_rate, gsl_rng *rng):
#
#     count = BIN_dt(N, discrete_rate, rng)
#     return count