import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultNu(BaseSetupEpi):
    """
    """
    nu = xs.foreign(BaseSEIR, 'nu', intent='out')

    def initialize(self):
        self.nu = self.get_nu()

    def get_nu(self):
        data = [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128]
        return xr.DataArray(
            data=data,
            dims=['age_group'],
            coords={k: self.COUNTS_COORDS[k] for k in dims}
        )
