import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from .seir import BruteForceSEIR
from ..setup.coords import InitDefaultCoords
from ..setup.phi import InitPhi, InitPhiGrpMapping


@xs.process
class BruteForceFOI:
    """A readable, brute force algorithm for calculating force of infection (FOI).
    """

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    phi_t = xs.foreign(InitPhi, 'phi_t', intent='in')
    phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    foi = xs.foreign(BruteForceSEIR, 'foi', intent='out')

    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')

    def calculate_foi(self) -> float:
        """
        """
        # assert isinstance(self.counts, xr.DataArray), type(self.counts)
        # assert isinstance(self.omega, xr.DataArray), type(self.omega)
        foi = 0.

        # Iterate over each vertex
        for v in range(self.counts.coords['vertex'].size):
            # Iterate over every pair of age-risk categories
            for a1, r1, a2, r2 in product(*[self.age_group, self.risk_group] * 2):
                if a1 == a2 and r1 == r2:
                    continue
                # logging.debug([a1, r1, a2, r2])

                age_pop = self.counts.loc[dict(
                    vertex=v, age_group=a2, # risk_group=r2
                )].sum(dim=['compartment', 'risk_group'])

                # Get the phi_grp indices
                phi_grp1 = self.phi_grp_mapping.loc[dict(
                    age_group=a1, risk_group=r1
                )]
                phi_grp2 = self.phi_grp_mapping.loc[dict(
                    age_group=a2, risk_group=r2
                )]

                # Get the value of phi
                phi = self.phi_t.loc[dict(
                    phi_grp1=phi_grp1, phi_grp2=phi_grp2
                )].values
                # logging.debug(f"phi: {phi}")

                # Get S compt
                counts_S = self.counts.loc[{
                    'vertex': v,
                    'age_group': a2,
                    'risk_group': r2,
                    'compartment': 'S'
                }].values
                # logging.debug(f"counts_S: {counts_S}")

                # Get value of beta
                # beta = self.beta.loc[{
                    # 'vertex': v,
                    # 'age_group': a2,
                    # 'risk_group': r2,
                    # 'compartment': 'S'
                # }]
                beta = self.beta
                assert isinstance(beta, Number)
                # logging.debug(f"beta: {beta}")

                # Get infectious compartments
                compt_I = ['Ia', 'Iy', 'Pa', 'Py']
                counts_I = self.counts.loc[{
                    'vertex': v,
                    'age_group': a2,
                    'risk_group': r2,
                    'compartment': compt_I
                }]
                # logging.debug(f"counts_I: {counts_I}")

                # Get value of omega for these infectious compartments
                # omega_I = self.omega.loc[{'age_group': a2, 'compartment': compt_I}]
                omega_I = 0.66

                # Calculate force of infection
                common_term = beta * phi * counts_S / age_pop
                # logging.debug(f"common_term: {common_term}")
                _sum = (common_term * omega_I * counts_I).sum(dim='compartment').values
                foi += _sum
        return foi


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
