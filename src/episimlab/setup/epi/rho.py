import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultRho(BaseSetupEpi):
    """
    """

    rho = xs.foreign(BaseSEIR, 'rho', intent='out')

    def initialize(self):
        self.rho = self.get_rho()

    def get_rho(self):
        dims = ['age_group', 'compartment']
        return xr.DataArray(
            data=0.43478261,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
