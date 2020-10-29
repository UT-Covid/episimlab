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

def brute_force_FOI(np.ndarray phi_grp_mapping,
                    np.ndarray counts,
                    np.ndarray phi_t,
                    # np.ndarray rho,
                    # np.ndarray gamma,
                    # np.ndarray pi,
                    # np.ndarray nu,
                    np.ndarray omega,
                    # float mu,
                    # float sigma,
                    # float eta,
                    # float tau,
                    float beta):
    """
    """
    cdef:
        long [:, :] phi_grp_view = phi_grp_mapping
        double [:, :, :, :] counts_view = counts
        double [:, :] phi_view = phi_t
        # double [:, :] rho_view = rho
        # double [:] gamma_view = gamma
        # double [:, :] pi_view = pi
        # double [:] nu_view = nu
        double [:, :] omega_view = omega
        # double [:] beta_view = beta
        # double [:] tau_view = tau
        # double [:] sigma_view = sigma
        # double [:] eta_view = eta

    return _brute_force_FOI(
        phi_grp_view,
        counts_view,
        phi_view,
        # rho_view,
        # gamma_view,
        # pi_view,
        # nu_view,
        omega_view,
        # floats
        # mu,
        # sigma,
        # eta,
        # tau,
        beta,
    )


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


cdef np.ndarray _brute_force_FOI(long [:, :] phi_grp_view,
                                double [:, :, :, :] counts_view,
                                double [:, :] phi_view,
                                # double [:, :] rho_view,
                                # double [:] gamma_view,
                                # double [:, :] pi_view,
                                # double [:] nu_view,
                                # age, compt
                                double [:, :] omega_view,
                                # double mu,
                                # double sigma,
                                # double eta,
                                # double tau,
                                double beta):
                                # double int_per_day):
                                # gsl_rng *rng):
    """
    TODO: pass all args
    TODO: return the deltas, not the updated counts
    """
    cdef:
        # DEBUG
        double int_per_day = 1.
        # indexers and lengths of each dimension in state space
        Py_ssize_t node_len = counts_view.shape[0]
        Py_ssize_t age_len = counts_view.shape[1]
        Py_ssize_t risk_len = counts_view.shape[2]
        Py_ssize_t compt_len = counts_view.shape[3]
        Py_ssize_t n, a, r, a_2, r_2

        np.ndarray foi = np.nan * np.empty(
            (node_len, age_len, risk_len), dtype=DTYPE_FLOAT)
        double [:, :, :] foi_view = foi
        # node population is the sum of all compartments for a given
        # node, age, risk, compartment
        np.ndarray node_pop_arr = np.sum(counts_view[:, :, :, :], axis=(2, -1))
        double [:, :] node_pop = node_pop_arr
        # epi params
        double gamma_a, gamma_y, gamma_h, nu, pi, \
            kappa, report_rate, rho_a, rho_y
        # epi params for force of infection calculation
        double beta_2, phi_1_2, omega_e_2, omega_pa_2, omega_py_2, omega_a_2, \
            omega_y_2, common_term, deterministic
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
        double rate_S2E
        # TODO
        double beta0 = beta

    # Iterate over node, age, and risk
    for n in prange(node_len, nogil=True):
    # for n in range(node_len):
        for a in range(age_len):
            for r in range(risk_len):

                # --------------   Expand epi parameters  --------------

                # 'count', 'beta0', 'sigma', 'gamma', 'eta', 'mu', 'omega', 'tau', 'nu', 'pi', 'kappa', 'report_rate', 'rho'
                # WARNING: index on 0 at compartments
                # dimension assumes that epi param
                # is same for all compartments
                # beta0 = counts_view[n, a, r, 1, 0]
                # sigma = discrete_time_approx(sigma, int_per_day)
                # gamma_a = discrete_time_approx(gamma_view[4], int_per_day)
                # gamma_y = discrete_time_approx(gamma_view[5], int_per_day)
                # gamma_h = discrete_time_approx(gamma_view[6], int_per_day)
                # eta = discrete_time_approx(eta, int_per_day)
                # mu = discrete_time_approx(mu, int_per_day)
                # _tau = counts_view[n, a, r, 7, 0]
                # nu = nu_view[a]
                # pi = pi_view[r, a]
                # _kappa = counts_view[n, a, r, 10, 0]
                # _report_rate = counts_view[n, a, r, 11, 0]
                # rho_a = discrete_time_approx(rho_view[a, 4], int_per_day)
                # rho_y = discrete_time_approx(rho_view[a, 5], int_per_day)
                # TODO: reimplement
                # _deterministic = counts_view[n, a, r, 13, 0]

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

                # -----------   Calculate force of infection  ----------

                rate_S2E = 0.
                for a_2 in range(age_len):
                    for r_2 in range(risk_len):

                        # Get phi
                        phi_1_2 = phi_view[phi_grp_view[a, r], phi_grp_view[a_2, r_2]]

                        beta_2 = beta
                        omega_a_2 = omega_view[a_2, 4]
                        omega_y_2 = omega_view[a_2, 5]
                        omega_pa_2 = omega_view[a_2, 2]
                        omega_py_2 = omega_view[a_2, 3]

                        # get compartments for a_2, r_2
                        Pa_2 = counts_view[n, a_2, r_2, 2]
                        Py_2 = counts_view[n, a_2, r_2, 3]
                        Ia_2 = counts_view[n, a_2, r_2, 4]
                        Iy_2 = counts_view[n, a_2, r_2, 5]

                        # Ignore case where node population is zero or negative
                        if node_pop[n, a_2] <= 0:
                            continue

                        # calculate force of infection
                        common_term = beta_2 * phi_1_2 * S / node_pop[n, a_2]
                        rate_S2E = rate_S2E + (common_term * (
                            (omega_a_2 * Ia_2) + \
                            (omega_y_2 * Iy_2) + \
                            (omega_pa_2 * Pa_2) + \
                            (omega_py_2 * Py_2)))

                foi_view[n, a, r] = rate_S2E
    return foi
