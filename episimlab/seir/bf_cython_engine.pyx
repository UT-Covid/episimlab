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
from ..cy_utils.cy_utils cimport get_seeded_rng, discrete_time_approx

# Abbreviate numpy dtypes
DTYPE_FLOAT = np.float64
DTYPE_INT = np.intc

# isinf from C math.h
cdef extern from "math.h" nogil:
    unsigned int isinf(double f)

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


def brute_force_SEIR(np.ndarray counts,
                     np.ndarray foi,
                     np.ndarray rho,
                     np.ndarray gamma,
                     np.ndarray pi,
                     np.ndarray nu,
                     float mu,
                     float sigma,
                     float eta,
                     float tau,
                     float int_per_day,
                     unsigned int stochastic,
                     unsigned int int_seed
                     ):
    """
    """
    cdef:
        double [:, :, :, :] counts_view = counts
        double [:] rho_view = rho
        double [:] gamma_view = gamma
        double [:, :] pi_view = pi
        double [:] nu_view = nu
        double [:, :, :] foi_view = foi
        # double [:] tau_view = tau
        # double [:] sigma_view = sigma
        # double [:] eta_view = eta
        # GSL random number generator
        gsl_rng *rng = get_seeded_rng(int_seed)

    return _brute_force_SEIR(
        counts_view,
        foi_view,
        rho_view,
        gamma_view,
        pi_view,
        nu_view,
        # floats
        mu,
        sigma,
        eta,
        tau,
        int_per_day,
        stochastic,
        rng
    )


cdef np.ndarray _brute_force_SEIR(double [:, :, :, :] counts_view,
                                  double [:, :, :] foi_view,
                                  double [:] rho_view,
                                  double [:] gamma_view,
                                  # risk, age
                                  double [:, :] pi_view,
                                  # age
                                  double [:] nu_view,
                                  # age, compt
                                  double mu,
                                  double sigma,
                                  double eta,
                                  double tau,
                                  double int_per_day,
                                  unsigned int stochastic,
                                  gsl_rng *rng,
                                  ):
                                  # double int_per_day):
    """
    TODO: clean up cdefs
    """
    cdef:
        # indexers and lengths of each dimension in state space
        Py_ssize_t node_len = counts_view.shape[0]
        Py_ssize_t age_len = counts_view.shape[1]
        Py_ssize_t risk_len = counts_view.shape[2]
        Py_ssize_t compt_len = counts_view.shape[3]
        Py_ssize_t n, a, r, a_2, r_2

        # output state array. Note that the only 'value_type' we are about
        # is index 0, or 'count'
        np.ndarray compt_counts = np.nan * np.empty(
            (node_len, age_len, risk_len, compt_len), dtype=DTYPE_FLOAT)
        double [:, :, :, :] compt_v = compt_counts
        # node population is the sum of all compartments for a given
        # node, age, risk, compartment
        np.ndarray node_pop_arr = np.sum(counts_view[:, :, :, :], axis=(2, -1))
        double [:, :] node_pop = node_pop_arr
        # epi params
        double gamma_a, gamma_y, gamma_h, nu, pi, \
            kappa, report_rate, rho_a, rho_y
        # epi params for force of infection calculation
        double deterministic
        # compartment counts
        double S, E, Pa, Py, Ia, Iy, Ih, R, D, E2P, E2Py, P2I, Pa2Ia, Py2Iy, \
            Iy2Ih, H2D
        # compartment counts for force of infection calculation
        double E_2, Ia_2, Iy_2, Pa_2, Py_2
        # delta compartment values
        double d_S, d_E, d_Pa, d_Py, d_Ia, d_Iy, d_Ih, d_R, d_D
        # new compartment values, after deltas applied
        double new_S, new_E, new_Pa, new_Py, new_Ia, new_Iy, new_Ih, new_R, \
            new_D, new_E2P, new_E2Py, new_P2I, new_Pa2Ia, new_Py2Iy, \
            new_Iy2Ih, new_H2D
        # rates between compartments
        double rate_S2E, rate_E2P, rate_Pa2Ia, rate_Py2Iy, rate_Ia2R, \
            rate_Iy2R, rate_Ih2R, rate_Iy2Ih, rate_Ih2D

    # Iterate over node, age, and risk
    for n in prange(node_len, nogil=True):
    # for n in range(node_len):
        for a in range(age_len):
            for r in range(risk_len):

                # -----------   Expand compartment counts  -------------

                # 'S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia', 'Py2Iy', 'Iy2Ih', 'H2D'
                S = counts_view[n, a, r, 0]
                E = counts_view[n, a, r, 1]
                Pa = counts_view[n, a, r, 2]
                Py = counts_view[n, a, r, 3]
                Ia = counts_view[n, a, r, 4]
                Iy = counts_view[n, a, r, 5]
                Ih = counts_view[n, a, r, 6]
                R = counts_view[n, a, r, 7]
                D = counts_view[n, a, r, 8]

                E2P = counts_view[n, a, r, 9]
                E2Py = counts_view[n, a, r, 10]
                P2I = counts_view[n, a, r, 11]
                Pa2Ia = counts_view[n, a, r, 12]
                Py2Iy = counts_view[n, a, r, 13]
                Iy2Ih = counts_view[n, a, r, 14]
                H2D = counts_view[n, a, r, 15]

                # --------------   Expand epi parameters  --------------

                # NOTE: be careful of variable name reuse here!
                gamma_a = discrete_time_approx(gamma_view[4], int_per_day)
                gamma_y = discrete_time_approx(gamma_view[5], int_per_day)
                gamma_h = discrete_time_approx(gamma_view[6], int_per_day)
                nu = nu_view[a]
                pi = pi_view[r, a]

                # ----------------   Get other deltas  -----------------

                rate_S2E = foi_view[n, a, r]
                rate_E2P = E * discrete_time_approx(sigma, int_per_day)
                rate_Pa2Ia = Pa * discrete_time_approx(rho_view[4], int_per_day)
                rate_Py2Iy = Py * discrete_time_approx(rho_view[5], int_per_day)
                rate_Ia2R = Ia * gamma_a
                rate_Iy2R = (1 - pi) * gamma_y * Iy
                rate_Ih2R = (1 - nu) * gamma_h * Ih
                rate_Iy2Ih = pi * Iy * discrete_time_approx(eta, int_per_day)
                rate_Ih2D = nu * Ih * discrete_time_approx(mu, int_per_day)

                # --------------   Sample from Poisson  ----------------

                if stochastic == 1:
                    rate_S2E = gsl_ran_poisson(rng, rate_S2E)
                    rate_E2P = gsl_ran_poisson(rng, rate_E2P)
                    rate_Py2Iy = gsl_ran_poisson(rng, rate_Py2Iy)
                    rate_Pa2Ia = gsl_ran_poisson(rng, rate_Pa2Ia)
                    rate_Ia2R = gsl_ran_poisson(rng, rate_Ia2R)
                    rate_Iy2R = gsl_ran_poisson(rng, rate_Iy2R)
                    rate_Ih2R = gsl_ran_poisson(rng, rate_Ih2R)
                    rate_Iy2Ih = gsl_ran_poisson(rng, rate_Iy2Ih)
                    rate_Ih2D = gsl_ran_poisson(rng, rate_Ih2D)

                if isinf(rate_S2E):
                    rate_S2E = 0
                if isinf(rate_E2P):
                    rate_E2P = 0
                if isinf(rate_Py2Iy):
                    rate_Py2Iy = 0
                if isinf(rate_Pa2Ia):
                    rate_Pa2Ia = 0
                if isinf(rate_Ia2R):
                    rate_Ia2R = 0
                if isinf(rate_Iy2R):
                    rate_Iy2R = 0
                if isinf(rate_Ih2R):
                    rate_Ih2R = 0
                if isinf(rate_Iy2Ih):
                    rate_Iy2Ih = 0
                if isinf(rate_Ih2D):
                    rate_Ih2D = 0

                # -----------------   Apply deltas  --------------------

                d_S = -rate_S2E
                new_S = S + d_S
                if new_S < 0:
                    rate_S2E = S
                    rate_S2E = 0

                d_E = rate_S2E - rate_E2P
                new_E = E + d_E
                if new_E < 0:
                    rate_E2P = E + rate_S2E
                    new_E = 0

                new_E2P = rate_E2P
                new_E2Py = tau * rate_E2P
                if new_E2Py < 0:
                    rate_E2P = 0
                    new_E2P = 0
                    new_E2Py = 0

                d_Pa = (1 - tau) * rate_E2P - rate_Pa2Ia
                new_Pa = Pa + d_Pa
                new_Pa2Ia = rate_Pa2Ia
                if new_Pa < 0:
                    rate_Pa2Ia = Pa + (1 - tau) * rate_E2P
                    new_Pa = 0
                    new_Pa2Ia = rate_Pa2Ia

                d_Py = tau * rate_E2P - rate_Py2Iy
                new_Py = Py + d_Py
                new_Py2Iy = rate_Py2Iy
                if new_Py < 0:
                    rate_Py2Iy = Py + tau * rate_E2P
                    new_Py = 0
                    new_Py2Iy = rate_Py2Iy

                new_P2I = new_Pa2Ia + new_Py2Iy

                d_Ia = rate_Pa2Ia - rate_Ia2R
                new_Ia = Ia + d_Ia
                if new_Ia < 0:
                    rate_Ia2R = Ia + rate_Pa2Ia
                    new_Ia = 0

                d_Iy = rate_Py2Iy - rate_Iy2R - rate_Iy2Ih
                new_Iy = Iy + d_Iy
                if new_Iy < 0:
                    rate_Iy2R = (Iy + rate_Py2Iy) * rate_Iy2R / \
                        (rate_Iy2R + rate_Iy2Ih)
                    rate_Iy2Ih = Iy + rate_Py2Iy - rate_Iy2R
                    new_Iy = 0

                new_Iy2Ih = rate_Iy2Ih
                if new_Iy2Ih < 0:
                    new_Iy2Ih = 0

                d_Ih = rate_Iy2Ih - rate_Ih2R - rate_Ih2D
                new_Ih = Ih + d_Ih
                if new_Ih < 0:
                    rate_Ih2R = (Ih + rate_Iy2Ih) * rate_Ih2R / \
                        (rate_Ih2R + rate_Ih2D)
                    rate_Ih2D = Ih + rate_Iy2Ih - rate_Ih2R
                    new_Ih = 0

                d_R = rate_Ia2R + rate_Iy2R + rate_Ih2R
                new_R = R + d_R

                d_D = rate_Ih2D
                new_H2D = rate_Ih2D
                new_D = D + d_D

                # ----------   Load new vals to state array  ---------------

                # 'S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia', 'Py2Iy', 'Iy2Ih', 'H2D'
                compt_v[n, a, r, 0] = new_S - S
                compt_v[n, a, r, 1] = new_E - E
                compt_v[n, a, r, 2] = new_Pa - Pa
                compt_v[n, a, r, 3] = new_Py - Py
                compt_v[n, a, r, 4] = new_Ia - Ia
                compt_v[n, a, r, 5] = new_Iy - Iy
                compt_v[n, a, r, 6] = new_Ih - Ih
                compt_v[n, a, r, 7] = new_R - R
                compt_v[n, a, r, 8] = new_D - D
    return compt_counts


def seir_with_foi(np.ndarray counts,
                     np.ndarray phi_t,
                     np.ndarray rho,
                     np.ndarray gamma,
                     np.ndarray pi,
                     np.ndarray nu,
                     np.ndarray omega,
                     float mu,
                     float beta,
                     float sigma,
                     float eta,
                     float tau,
                     float int_per_day,
                     unsigned int stochastic,
                     unsigned int int_seed
                     ):
    """
    """
    cdef:
        double [:, :, :, :] counts_view = counts
        double [:, :, :, :, :, :] phi_view = phi_t
        double [:] rho_view = rho
        double [:] gamma_view = gamma
        double [:, :] pi_view = pi
        double [:, :] omega_view = omega
        double [:] nu_view = nu
        # GSL random number generator
        gsl_rng *rng = get_seeded_rng(int_seed)

    return _seir_with_foi(
        counts_view,
        phi_view,
        rho_view,
        gamma_view,
        pi_view,
        nu_view,
        omega_view,
        # floats
        mu,
        beta,
        sigma,
        eta,
        tau,
        int_per_day,
        stochastic,
        rng
    )


cdef np.ndarray _seir_with_foi(double [:, :, :, :] counts_view,
                                double [:, :, :, :, :, :] phi_view,
                                  double [:] rho_view,
                                  double [:] gamma_view,
                                  # risk, age
                                  double [:, :] pi_view,
                                  # age
                                  double [:] nu_view,
                                  # risk, age
                                  double [:, :] omega_view,
                                  # age, compt
                                  double mu,
                                  double beta,
                                  double sigma,
                                  double eta,
                                  double tau,
                                  double int_per_day,
                                  unsigned int stochastic,
                                  gsl_rng *rng,
                                  ):
    """
    TODO: clean up cdefs
    """
    cdef:
        # indexers and lengths of each dimension in state space
        Py_ssize_t node_len = counts_view.shape[0]
        Py_ssize_t age_len = counts_view.shape[1]
        Py_ssize_t risk_len = counts_view.shape[2]
        Py_ssize_t compt_len = counts_view.shape[3]
        Py_ssize_t n, a, r, n_2, a_2, r_2

        double phi_1_2, omega_e_2, omega_pa_2, omega_py_2, omega_a_2, \
            omega_y_2, common_term, deterministic
        # output state array. Note that the only 'value_type' we are about
        # is index 0, or 'count'
        np.ndarray compt_counts = np.nan * np.empty(
            (node_len, age_len, risk_len, compt_len), dtype=DTYPE_FLOAT)
        double [:, :, :, :] compt_v = compt_counts
        # node population is the sum of all compartments for a given
        # node, age, risk, compartment
        np.ndarray total_pop_arr = np.sum(counts_view[:, :, :, :], axis=(-1))
        double [:, :, :] total_pop = total_pop_arr
        np.ndarray node_pop_arr = np.sum(counts_view[:, :, :, :], axis=(2, -1))
        double [:, :] node_pop = node_pop_arr
        # epi params
        double gamma_a, gamma_y, gamma_h, nu, pi, \
            kappa, report_rate, rho_a, rho_y
        # compartment counts
        double S, E, Pa, Py, Ia, Iy, Ih, R, D, E2P, E2Py, P2I, Pa2Ia, Py2Iy, \
            Iy2Ih, H2D
        # compartment counts for force of infection calculation
        double E_2, Ia_2, Iy_2, Pa_2, Py_2
        # delta compartment values
        double d_S, d_E, d_Pa, d_Py, d_Ia, d_Iy, d_Ih, d_R, d_D
        # new compartment values, after deltas applied
        double new_S, new_E, new_Pa, new_Py, new_Ia, new_Iy, new_Ih, new_R, \
            new_D, new_E2P, new_E2Py, new_P2I, new_Pa2Ia, new_Py2Iy, \
            new_Iy2Ih, new_H2D
        # rates between compartments
        double rate_S2E, rate_E2P, rate_Pa2Ia, rate_Py2Iy, rate_Ia2R, \
            rate_Iy2R, rate_Ih2R, rate_Iy2Ih, rate_Ih2D

    # Iterate over node, age, and risk
    for n in prange(node_len, nogil=True):
    # for n in range(node_len):
        for a in range(age_len):
            for r in range(risk_len):

                # -----------   Expand compartment counts  -------------

                # 'S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia', 'Py2Iy', 'Iy2Ih', 'H2D'
                S = counts_view[n, a, r, 0]
                E = counts_view[n, a, r, 1]
                Pa = counts_view[n, a, r, 2]
                Py = counts_view[n, a, r, 3]
                Ia = counts_view[n, a, r, 4]
                Iy = counts_view[n, a, r, 5]
                Ih = counts_view[n, a, r, 6]
                R = counts_view[n, a, r, 7]
                D = counts_view[n, a, r, 8]

                E2P = counts_view[n, a, r, 9]
                E2Py = counts_view[n, a, r, 10]
                P2I = counts_view[n, a, r, 11]
                Pa2Ia = counts_view[n, a, r, 12]
                Py2Iy = counts_view[n, a, r, 13]
                Iy2Ih = counts_view[n, a, r, 14]
                H2D = counts_view[n, a, r, 15]

                # --------------   Expand epi parameters  --------------

                # NOTE: be careful of variable name reuse here!
                gamma_a = discrete_time_approx(gamma_view[4], int_per_day)
                gamma_y = discrete_time_approx(gamma_view[5], int_per_day)
                gamma_h = discrete_time_approx(gamma_view[6], int_per_day)
                nu = nu_view[a]
                pi = pi_view[r, a]

                # --------------   Calculate FOI  --------------

                rate_S2E = 0.
                S = counts_view[n, a, r, 0]
                for n_2 in range(node_len):
                    for a_2 in range(age_len):
                        for r_2 in range(risk_len):
                            # Ignore case where node population is zero or negative
                            if total_pop[n_2, a_2, r_2] <= 0:
                                continue

                            # Get phi
                            phi_1_2 = phi_view[n, n_2, a, a_2, r, r_2]

                            # Enumerate omega
                            omega_a_2 = omega_view[a_2, 4]
                            omega_y_2 = omega_view[a_2, 5]
                            omega_pa_2 = omega_view[a_2, 2]
                            omega_py_2 = omega_view[a_2, 3]

                            # Get compartments for a_2, r_2
                            Pa_2 = counts_view[n_2, a_2, r_2, 2]
                            Py_2 = counts_view[n_2, a_2, r_2, 3]
                            Ia_2 = counts_view[n_2, a_2, r_2, 4]
                            Iy_2 = counts_view[n_2, a_2, r_2, 5]

                            # calculate force of infection
                            common_term = beta * phi_1_2 * S / total_pop[n_2, a_2, r_2]
                            rate_S2E = rate_S2E + (common_term * (
                                (omega_a_2 * Ia_2) + \
                                (omega_y_2 * Iy_2) + \
                                (omega_pa_2 * Pa_2) + \
                                (omega_py_2 * Py_2)))

                # ----------------   Get other deltas  -----------------

                rate_E2P = E * discrete_time_approx(sigma, int_per_day)
                rate_Pa2Ia = Pa * discrete_time_approx(rho_view[4], int_per_day)
                rate_Py2Iy = Py * discrete_time_approx(rho_view[5], int_per_day)
                rate_Ia2R = Ia * gamma_a
                rate_Iy2R = (1 - pi) * gamma_y * Iy
                rate_Ih2R = (1 - nu) * gamma_h * Ih
                rate_Iy2Ih = pi * Iy * discrete_time_approx(eta, int_per_day)
                rate_Ih2D = nu * Ih * discrete_time_approx(mu, int_per_day)

                # --------------   Sample from Poisson  ----------------

                if stochastic == 1:
                    rate_S2E = gsl_ran_poisson(rng, rate_S2E)
                    rate_E2P = gsl_ran_poisson(rng, rate_E2P)
                    rate_Py2Iy = gsl_ran_poisson(rng, rate_Py2Iy)
                    rate_Pa2Ia = gsl_ran_poisson(rng, rate_Pa2Ia)
                    rate_Ia2R = gsl_ran_poisson(rng, rate_Ia2R)
                    rate_Iy2R = gsl_ran_poisson(rng, rate_Iy2R)
                    rate_Ih2R = gsl_ran_poisson(rng, rate_Ih2R)
                    rate_Iy2Ih = gsl_ran_poisson(rng, rate_Iy2Ih)
                    rate_Ih2D = gsl_ran_poisson(rng, rate_Ih2D)

                if isinf(rate_S2E):
                    rate_S2E = 0
                if isinf(rate_E2P):
                    rate_E2P = 0
                if isinf(rate_Py2Iy):
                    rate_Py2Iy = 0
                if isinf(rate_Pa2Ia):
                    rate_Pa2Ia = 0
                if isinf(rate_Ia2R):
                    rate_Ia2R = 0
                if isinf(rate_Iy2R):
                    rate_Iy2R = 0
                if isinf(rate_Ih2R):
                    rate_Ih2R = 0
                if isinf(rate_Iy2Ih):
                    rate_Iy2Ih = 0
                if isinf(rate_Ih2D):
                    rate_Ih2D = 0

                # -----------------   Apply deltas  --------------------

                d_S = -rate_S2E
                new_S = S + d_S
                if new_S < 0:
                    rate_S2E = S
                    rate_S2E = 0

                d_E = rate_S2E - rate_E2P
                new_E = E + d_E
                if new_E < 0:
                    rate_E2P = E + rate_S2E
                    new_E = 0

                new_E2P = rate_E2P
                new_E2Py = tau * rate_E2P
                if new_E2Py < 0:
                    rate_E2P = 0
                    new_E2P = 0
                    new_E2Py = 0

                d_Pa = (1 - tau) * rate_E2P - rate_Pa2Ia
                new_Pa = Pa + d_Pa
                new_Pa2Ia = rate_Pa2Ia
                if new_Pa < 0:
                    rate_Pa2Ia = Pa + (1 - tau) * rate_E2P
                    new_Pa = 0
                    new_Pa2Ia = rate_Pa2Ia

                d_Py = tau * rate_E2P - rate_Py2Iy
                new_Py = Py + d_Py
                new_Py2Iy = rate_Py2Iy
                if new_Py < 0:
                    rate_Py2Iy = Py + tau * rate_E2P
                    new_Py = 0
                    new_Py2Iy = rate_Py2Iy

                new_P2I = new_Pa2Ia + new_Py2Iy

                d_Ia = rate_Pa2Ia - rate_Ia2R
                new_Ia = Ia + d_Ia
                if new_Ia < 0:
                    rate_Ia2R = Ia + rate_Pa2Ia
                    new_Ia = 0

                d_Iy = rate_Py2Iy - rate_Iy2R - rate_Iy2Ih
                new_Iy = Iy + d_Iy
                if new_Iy < 0:
                    rate_Iy2R = (Iy + rate_Py2Iy) * rate_Iy2R / \
                        (rate_Iy2R + rate_Iy2Ih)
                    rate_Iy2Ih = Iy + rate_Py2Iy - rate_Iy2R
                    new_Iy = 0

                new_Iy2Ih = rate_Iy2Ih
                if new_Iy2Ih < 0:
                    new_Iy2Ih = 0

                d_Ih = rate_Iy2Ih - rate_Ih2R - rate_Ih2D
                new_Ih = Ih + d_Ih
                if new_Ih < 0:
                    rate_Ih2R = (Ih + rate_Iy2Ih) * rate_Ih2R / \
                        (rate_Ih2R + rate_Ih2D)
                    rate_Ih2D = Ih + rate_Iy2Ih - rate_Ih2R
                    new_Ih = 0

                d_R = rate_Ia2R + rate_Iy2R + rate_Ih2R
                new_R = R + d_R

                d_D = rate_Ih2D
                new_H2D = rate_Ih2D
                new_D = D + d_D

                # ----------   Load new vals to state array  ---------------

                # 'S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia', 'Py2Iy', 'Iy2Ih', 'H2D'
                compt_v[n, a, r, 0] = new_S - S
                compt_v[n, a, r, 1] = new_E - E
                compt_v[n, a, r, 2] = new_Pa - Pa
                compt_v[n, a, r, 3] = new_Py - Py
                compt_v[n, a, r, 4] = new_Ia - Ia
                compt_v[n, a, r, 5] = new_Iy - Iy
                compt_v[n, a, r, 6] = new_Ih - Ih
                compt_v[n, a, r, 7] = new_R - R
                compt_v[n, a, r, 8] = new_D - D
    return compt_counts
