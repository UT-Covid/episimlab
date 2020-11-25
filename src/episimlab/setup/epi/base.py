import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..coords import InitDefaultCoords
from ...apply_counts_delta import ApplyCountsDelta


@xs.process
class BaseSetupEpi:
    """
    """
    COUNTS_DIMS = ApplyCountsDelta.COUNTS_DIMS

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')
    compartment = xs.foreign(InitDefaultCoords, 'compartment', intent='in')

    @property
    def counts_coords(self):
        return {dim: getattr(self, dim) for dim in self.COUNTS_DIMS}
