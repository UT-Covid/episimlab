import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi
from .symp_h_ratio import SetupDefaultSympHRatioWithRisk


@xs.process
class SetupDefaultPi(BaseSetupEpi):
    """
    """
    pi = xs.foreign(BaseSEIR, 'pi', intent='out')

    def initialize(self):
        self.pi = self.get_pi()

    def get_pi(self):
        data = np.array([
            [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
            [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]
        ])
        dims = ['risk_group', 'age_group']
        coords = {k: self.counts_coords[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)


@xs.process
class SetupStaticPi(SetupDefaultPi):
    """Calculate pi after sampling once from this triangular distibution,
    at the beginning of the simulation.
    """
    symp_h_ratio_w_risk = xs.foreign(SetupDefaultSympHRatioWithRisk,
                                     'symp_h_ratio_w_risk', intent='in')
    gamma = xs.foreign(BaseSEIR, 'gamma', intent='in')
    eta = xs.foreign(BaseSEIR, 'eta', intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    def get_pi(self) -> xr.DataArray:
        dims = ['risk_group', 'age_group']
        gamma_y = float(self.gamma.loc[dict(compartment='Iy')])
        da = ((self.symp_h_ratio_w_risk * gamma_y) /
              (self.eta + (gamma_y - self.eta) * self.symp_h_ratio_w_risk))
        assert isinstance(da, xr.DataArray)
        assert all((dim in da.coords for dim in dims)), (da.coords, dims)
        return da


@xs.process
class SetupDynamicPi(SetupStaticPi):
    """Like SetupStaticPi, but recalculate at every step
    """

    def run_step(self):
        self.pi = self.get_pi()
