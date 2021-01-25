import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..setup.coords import InitDefaultCoords


@xs.process
class BaseFOI:
    """
    """
    FOI_DIMS = ('vertex', 'age_group', 'risk_group')

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')

    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')

    foi = xs.variable(dims=FOI_DIMS, intent='out')

    def initialize(self):
        self.foi = xr.DataArray(
            data=0.,
            dims=self.FOI_DIMS,
            coords={dim: getattr(self, dim) for dim in self.FOI_DIMS}
        )
