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
from ..cy_utils.cy_utils cimport get_seeded_rng

# Abbreviate numpy dtypes
DTYPE_FLOAT = np.float64
DTYPE_INT = np.intc

# Random generator from GSL lib
cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    void gsl_rng_free(gsl_rng * r)

# Poisson distribution from GSL lib
cdef extern from "gsl/gsl_randist.h" nogil:
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)


def graph_high_gran(np.ndarray counts,
                    np.ndarray adj_t,
                    np.ndarray adj_grp_mapping,
                    unsigned int stochastic,
                    unsigned int int_seed
                    ):
    """
    """
    cdef:
        double [:, :, :, :] counts_view = counts
        double [:, :] adj_view = adj_t
        long [:, :, :, :] mapping_view = adj_grp_mapping
        # GSL random number generator
        gsl_rng *rng = get_seeded_rng(int_seed)

    return _graph_high_gran(counts_view, adj_view, mapping_view, stochastic, rng)


cdef np.ndarray _graph_high_gran(double [:, :, :, :] counts_view,
                                 double [:, :] adj_view,
                                 long [:, :, :, :] mapping_view,
                                 unsigned int stochastic,
                                 gsl_rng *rng,
                                 ):
    """
    """

    # Type setting
    cdef:
        # lengths of each dimension in state space
        Py_ssize_t node_len = counts_view.shape[0]
        Py_ssize_t age_len = counts_view.shape[1]
        Py_ssize_t risk_len = counts_view.shape[2]
        Py_ssize_t compt_len = counts_view.shape[3]
        # output state array. Note that `value_type` dimension is
        # removed, since no coordinate other than `counts` is changing
        np.ndarray delta = np.zeros(
            (node_len, age_len, risk_len, compt_len), dtype=DTYPE_FLOAT)
        # counters
        Py_ssize_t c, c2, a, r, ct, idx1, idx2
        double c2_to_c
        double [:, :, :, :] d_view = delta

    # ------------------------------------------------------------------

    # Iterate over every pair of nodes
    for c in prange(node_len, nogil=True):
    # for c in range(node_len):
        for c2 in range(node_len):
            # No migration within node (`c` == `c2`), and ensure that
            # each unique pair of nodes is iterated only once.
            if c2 <= c:
                continue
            # Refresh the views on cities, to incorporate changes made
            # during previous iterations on node `c` or `c2`
            for a in range(age_len):
                for r in range(risk_len):
                    for ct in range(compt_len):
                        # Get adj indices from the adj_grp_mapping
                        idx1 = mapping_view[c, a, r, ct]
                        idx2 = mapping_view[c2, a, r, ct]

                        # For this compartment, net migration from c2 to c1
                        c2_to_c = (counts_view[c2, a, r, ct] * adj_view[idx1, idx2]) - \
                            (counts_view[c, a, r, ct] * adj_view[idx2, idx1])

                        # Handle stochasticity if specified
                        if stochastic == 1:
                            if c2_to_c < 0:
                                c2_to_c = gsl_ran_poisson(rng, -c2_to_c)
                                c2_to_c = -c2_to_c
                            else:
                                c2_to_c = gsl_ran_poisson(rng, c2_to_c)

                        # Ensure that no compartments will have negative
                        # values, while ensuring that the total sum of
                        # the delta array is zero
                        if (d_view[c, a, r, ct] + c2_to_c + counts_view[c, a, r, ct]) < 0:
                            # node `c` would be negative
                            # logging.error(f"new_delt_c: {new_delt_c}")
                            c2_to_c = -counts_view[c, a, r, ct] - d_view[c, a, r, ct]
                        elif (d_view[c2, a, r, ct] - c2_to_c + counts_view[c2, a, r, ct]) < 0:
                            # node `c2` would be negative
                            # logging.error(f"new_delt_c2: {new_delt_c2}")
                            c2_to_c = counts_view[c2, a, r, ct] + d_view[c2, a, r, ct]

                        # Update the delta array
                        d_view[c, a, r, ct] += c2_to_c
                        d_view[c2, a, r, ct] -= c2_to_c
    return delta
