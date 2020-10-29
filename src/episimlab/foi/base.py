import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..setup import InitDefaultCoords

@xs.process
class BaseFOI:
    """
    """
    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')

    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')

    # TODO: needs a vertex dimension
    foi = xs.variable(dims=('age_group', 'risk_group'), intent='out')

    def initialize(self):
        self.foi = xr.DataArray(
            data=0.,
            dims=('age_group', 'risk_group'),
            coords=dict(
                age_group=self.age_group,
                risk_group=self.risk_group
            )
        )
