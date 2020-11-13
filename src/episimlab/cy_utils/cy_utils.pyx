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
        return (1 - (1 - rate)**(1/timestep))
