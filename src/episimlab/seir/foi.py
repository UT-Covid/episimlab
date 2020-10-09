import xsimlab as xs
import logging
from itertools import product


@xs.process
class BruteForceFOI:

    age_group = xs.variable()
    risk_group = xs.variable()
    phi_t = xs.variable(dims=('phi_grp1', 'phi_grp2'))
    beta = xs.variable()
    omega = xs.variable(dims=('compartment'))
    counts = xs.variable(
        dims=('vertex', 'age_group', 'risk_group', 'compartment'),
        static=False
    )
    foi = xs.variable(intent='out')

    def calculate_foi(self):
        """
        """
        # Iterate over every pair of age-risk categories
        for a1, r1, a2, r2 in product(*[self.age_group.values, self.risk_group.values] * 2):
            if a1 == a2 and r1 == r2:
                continue
            # logging.debug([a1, r1, a2, r2])


    def run_step(self):
        self.foi = self.calculate_foi()








def get_foi_numpy(compt_ia, compt_iy, compt_pa, compt_py, compt_s, phi_, beta, kappa,
            omega_a, omega_y, omega_pa, omega_py, age_pop, n_age, n_risk):
    """From SEIR-city v1.4 d0902c0af796a6ac3a15ec833ae24dcfa81d9f2b"""

    # reshape phi
    phi_tile = np.tile(phi_, (n_risk, n_risk))

    def _si_contact(s_compt, i_compt, omega):

        # transpositions are needed b/c the state arrays have rows = age, cols = risk,
        # but the parameters have rows = risk, cols = age
        it = i_compt.transpose()
        st = s_compt.transpose()
        si_outer = np.outer(st, it)
        si_omega = si_outer * omega.ravel()
        si_contact = si_omega * phi_tile
        si_beta = si_contact * beta.ravel()
        age_tile = np.tile(age_pop, (1, n_risk))
        if np.isinf(age_tile).any():
            raise ValueError('Tiled age structure array contains np.inf values')
        si_pop = si_beta/age_tile
        si_a1_r = si_pop.sum(axis=1).reshape(n_risk, n_age)

        return si_a1_r

    foi = np.zeros((n_risk, n_age))

    # from compt_ia
    foi += _si_contact(compt_s, compt_ia, omega_a)

    # from compt_iy
    foi += _si_contact(compt_s, compt_iy, omega_y)

    # from compt_pa
    foi += _si_contact(compt_s, compt_pa, omega_pa)

    # from compt_py
    foi += _si_contact(compt_s, compt_py, omega_py)

    return foi.transpose()
