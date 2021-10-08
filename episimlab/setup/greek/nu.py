import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..compt_model import ComptModel
from ..utils import group_dict_by_var, get_var_dims, trim_data_to_coords


@xs.process
class SetupStaticNu:
    """Calculate nu after sampling once from this triangular distibution,
    at the beginning of the simulation.
    """
    hosp_f_ratio = xs.variable(dims=('age_group'),
                               static=True, intent='in')
    gamma = xs.foreign(BaseSEIR, 'gamma', intent='in')
    mu = xs.foreign(BaseSEIR, 'mu', intent='in')

    def get_nu(self) -> xr.DataArray:
        dims = ['age_group']
        gamma_h = float(self.gamma.loc[dict(compartment='Ih')])
        da = (self.hosp_f_ratio * gamma_h /
              (self.mu + (gamma_h - self.mu) * self.hosp_f_ratio))
        assert isinstance(da, xr.DataArray), type(da)
        assert all((dim in da.coords for dim in dims)), (da.coords, dims)
        return da


@xs.process
class SetupDynamicNu(SetupStaticNu):
    """Like SetupStaticNu, but recalculate at every step.
    """

    def run_step(self):
        self.nu = self.get_nu()
