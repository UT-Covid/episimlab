import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from .base import VaccineFOI


@xs.process
class BruteForceVaccineFOI(VaccineFOI):
    """A readable, brute force algorithm for calculating force of infection (FOI) with vaccination.
    """

    phi_t = xs.global_ref('phi_t')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')

    def run_step(self):
        """
        """
        # Instantiate as array of zeros
        self.foi = xr.DataArray(
            data=0.,
            dims=self.FOI_DIMS,
            coords={dim: getattr(self, dim) for dim in self.FOI_DIMS}
        )

        # Iterate over every pair of unique vertex-age-risk combinations
        for v1, a1, r1, v2, a2, r2 in product(*[self.vertex, self.age_group,
                                                self.risk_group] * 2):
            total_pop = self.counts.loc[dict(
                vertex=v2, age_group=a2, risk_group=r2
            )].sum(dim=['compartment'])

            # Get the value of phi
            phi = self.phi_t.loc[dict(
                vertex1=v1,
                vertex2=v2,
                age_group1=a1,
                age_group2=a2,
                risk_group1=r1,
                risk_group2=r2,
            )].values

            # Get specified compt
            counts_S = self.counts.loc[{
                'vertex': v1,
                'age_group': a1,
                'risk_group': r1,
                'compartment': 'S'
            }].values

            counts_V = self.counts.loc[{
                'vertex': v1,
                'age_group': a1,
                'risk_group': r1,
                'compartment': 'V'
            }].values

            # Get value of beta
            beta = self.beta_arr[0]
            beta_vacc = self.beta_arr[1]
            assert isinstance(beta, Number)

            # Get infectious compartments
            compt_I = ['Ia', 'Iy', 'Pa', 'Py']
            counts_I = self.counts.loc[{
                'vertex': v2,
                'age_group': a2,
                'risk_group': r2,
                'compartment': compt_I
            }]

            # Get value of omega for these infectious compartments
            omega_I = self.omega.loc[{'age_group': a2, 'compartment': compt_I}]

            # Calculate force of infection
            common_term_S = beta * phi * counts_S / total_pop
            _sum = (common_term_S * omega_I * counts_I).sum(dim='compartment').values
            # gah this FOI array is a whole mood. need to add a dimension for compartment to differentiate S and V
            self.foi.loc[dict(vertex=v1, age_group=a1, risk_group=r1)] += _sum
