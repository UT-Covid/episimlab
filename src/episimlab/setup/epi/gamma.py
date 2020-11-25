import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultGamma(BaseSetupEpi):
    """
    """

    gamma = xs.foreign(BaseSEIR, 'gamma', intent='out')

    def initialize(self):
        self.gamma = self.get_gamma()

    def get_gamma(self):
        dims = ["compartment"]
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
        return da
