import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...foi.base import BaseFOI
from .base import BaseSetupEpi


@xs.process
class SetupDefaultOmega(BaseSetupEpi):
    """
    """
    omega = xs.foreign(BaseFOI, 'omega', intent='out')

    def initialize(self):
        self.omega = self.get_omega()

    def get_omega(self):
        # omega_a = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667])
        # omega_y = np.array([1.        , 1.        , 1.        , 1.        , 1.        ])
        # omega_h = np.array([0.        , 0.        , 0.        , 0.        , 0.        ])
        # omega_pa = np.array([0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149])
        # omega_py = np.array([1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724])
        data = np.array([[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
                         [1.        , 1.        , 1.        , 1.        , 1.        ],
                         [0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149],
                         [1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724]])

        dims = ['age_group', 'compartment']
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
        assert isinstance(da, xr.DataArray), type(da)
        return da
