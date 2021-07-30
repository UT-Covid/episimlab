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
from ..cy_utils.cy_utils cimport get_seeded_rng, discrete_time_approx, discrete_time_sigmoid_approx, calc_binom_pop_change

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

def seir_with_foi(np.ndarray counts,
                     np.ndarray phi_t,
                     np.ndarray rho,
                     np.ndarray gamma,
                     np.ndarray pi,
                     np.ndarray nu,
                     np.ndarray omega,
                     float mu,
                     float beta,
                     float beta_vacc
                     float sigma,
                     float eta,
                     float tau,
                     float tau_vacc,
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
        beta_vacc,
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
                                  double beta_vacc,
                                  double sigma,
                                  double eta,
                                  double tau,
                                  double tau_vacc,
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
            Iy2Ih, H2D, V, Ev
        # compartment counts for force of infection calculation
        double E_2, Ia_2, Iy_2, Pa_2, Py_2
        # delta compartment values
        double d_S, d_E, d_Pa, d_Py, d_Ia, d_Iy, d_Ih, d_R, d_D
        # new compartment values, after deltas applied
        double new_S, new_E, new_Pa, new_Py, new_Ia, new_Iy, new_Ih, new_R, \
            new_D, new_E2P, new_E2Py, new_P2I, new_Pa2Ia, new_Py2Iy, \
            new_Iy2Ih, new_H2D, newEv
        # rates between compartments
        double leaving_S_rate, rate_E2P, rate_Pa2Ia, rate_Py2Iy, rate_Ia2R, \
               rate_Iy2R, rate_Ih2R, rate_Iy2Ih, rate_Ih2D, leaving_E_rate

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

                # new vaccine-related compartments
                V = counts_view[n, a, r, 16]
                Ev = counts_view[n, a, r, 17]
                V2Ev = counts_view[n, a, r, 18]
                Ev2P = counts_view[n, a, r, 19]

                # --------------   Calculate FOI  --------------

                leaving_S_rate = 0.
                leaving_E_rate = 0.
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

                            # calculate force of infection on susceptible
                            common_term = beta * phi_1_2 * S / total_pop[n_2, a_2, r_2]
                            leaving_S_rate = leaving_S_rate + (common_term * (
                                (omega_a_2 * Ia_2) + \
                                (omega_y_2 * Iy_2) + \
                                (omega_pa_2 * Pa_2) + \
                                (omega_py_2 * Py_2)))

                            # calculate force of infection on vaccinated
                            common_term = beta_vacc * phi_1_2 * S / total_pop[n_2, a_2, r_2]
                            leaving_E_rate = leaving_E_rate + (common_term * (
                                    (omega_a_2 * Ia_2) + \
                                    (omega_y_2 * Iy_2) + \
                                    (omega_pa_2 * Pa_2) + \
                                    (omega_py_2 * Py_2)))

                # ----------------   Get other deltas  -----------------

                # TO DO: factory class for stochastic vs deterministic delta calculation under binomial transitions
                # why? deterministic N must be float. stochastic w/binomial: N must be unsigned int

                # S -> E
                new_E = calc_binom_pop_change(S, leaving_S_rate, stochastic)

                # V -> Ev, disregarding new vaccinations
                new_Ev = calc_binom_pop_change(V, leaving_E_rate, stochastic)

                # E -> P and Ev -> P
                discrete_sigma = discrete_time_sigmoid_approx(sigma, int_per_day)
                if stochastic == 1:
                    leaving_E = calc_binom_pop_change(E, discrete_sigma)
                    leaving_EV = calc_binom_pop_change(Ev, discrete_sigma)
                else:
                    leaving_E = E * discrete_sigma
                    leaving_EV = Ev * discrete_sigma

                # split P to Pa, Py (note: tau is a proportion, not a rate)
                if stochastic == 1:
                    E2Pa = calc_binom_pop_change(leaving_E, (1-tau))
                    Ev2Pa = calc_binom_pop_change(leaving_Ev, (1-tau_vacc))
                else:
                    E2Pa = leaving_E * (1-tau)
                    Ev2Pa = leaving_Ev * (1-tau)

                new_Pa =  E2Pa + Ev2Pa
                new_Py = (leaving_E + leaving_EV) - new_Pa

                # P -> I
                discrete_rhoA = discrete_time_sigmoid_approx(rho_view[4], int_per_day)
                discrete_rhoY = discrete_time_sigmoid_approx(rho_view[5], int_per_day)

                if stochastic == 1:
                    new_Ia = calc_binom_pop_change(Pa, discrete_rhoA)
                    new_Iy = calc_binom_pop_change(Py, discrete_rhoY)
                else:
                    new_Ia = Pa * discrete_rhoA
                    new_Iy = Py * discrete_rhoY

                # Ia -> R
                discrete_gammaA = discrete_time_sigmoid_approx(gamma_view[4], int_per_day)

                if stochastic == 1:
                    recovering_Ia = calc_binom_pop_change(Ia, discrete_gammaA)
                else:
                    recovering_Ia = Ia * discrete_gammaA

                # Iy -> R or IH
                pi = pi_view[r, a]
                leaving_Iy_rate = ((1.0 - pi) * gamma_Y + pi * eta)
                discrete_leaving_Iy = discrete_time_sigmoid_approx(leaving_Iy_rate, int_per_day)

                if stochastic == 1:
                    leaving_Iy = calc_binom_pop_change(Iy, discrete_leaving_Iy)
                else:
                    leaving_Iy = Iy * discrete_leaving_Iy

                # split leaving Iy to R and H
                if stochastic == 1:
                    new_Ih = calc_binom_pop_change(leaving_Iy, pi * eta / leaving_Iy_rate)
                else:
                    new_Ih = leaving_Iy * (pi * eta / leaving_Iy_rate)

                recovering_Iy = leaving_Iy - new_Ih

                # Ih -> R or D
                nu = nu_view[a]
                leaving_H_rate = nu * mu + (1.0 - nu) * gamma_h
                discrete_leaving_H_rate = discrete_time_sigmoid_approx(leaving_H_rate, int_per_day)

                if stochastic == 1:
                    leaving_H = calc_binom_pop_change(Ih, discrete_leaving_H_rate)
                    recovering_H = calc_binom_pop_change(leaving_H, (1 - nu) * gamma_H / leaving_H_rate)
                else:
                    leaving_H = Ih * discrete_leaving_H_rate
                    recovering_H = leaving_H * ((1 - nu) * gamma_H / leaving_H_rate)

                dying_H = leaving_H - recovering_H

                # S -> V ### TO DO ###
                new_V = 0.0

                # ----------   Load deltas to state array  ---------------

                # 'S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia', 'Py2Iy', 'Iy2Ih', 'H2D', 'V', 'Ev'
                compt_v[n, a, r, 0] = new_S
                compt_v[n, a, r, 1] = new_E
                compt_v[n, a, r, 2] = new_Pa
                compt_v[n, a, r, 3] = new_Py
                compt_v[n, a, r, 4] = new_Ia
                compt_v[n, a, r, 5] = new_Iy
                compt_v[n, a, r, 6] = new_Ih
                compt_v[n, a, r, 7] = new_R
                compt_v[n, a, r, 8] = new_D

                # new vaccine-related compartments
                compt_v[n, a, r, 16] = new_V - V
                compt_v[n, a, r, 17] = new_Ev - Ev

    return compt_counts
