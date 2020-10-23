import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..seir.base import BaseSEIR

@xs.process
class BaseFOI:
    """
    """
    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')
    # TODO: needs a vertex dimension
    foi = xs.foreign(BaseSEIR, 'foi', intent='out')

    def initialize(self):
        self.foi = 0.
