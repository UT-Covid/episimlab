import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


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
        dims = ('risk_group', 'age_group')
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)
