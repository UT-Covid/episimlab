import xsimlab as xs
import xarray as xr
from .coords import InitDefaultCoords

from ..apply_counts_delta import ApplyCountsDelta

@xs.process
class InitDefaultCounts:

    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='out')
    age_group = xs.foreign(InitDefaultCoords, 'age_group')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group')
    compartment = xs.foreign(InitDefaultCoords, 'compartment')
    vertex = xs.foreign(InitDefaultCoords, 'vertex')

    def initialize(self):
        self.counts = xr.DataArray(
            data=0.,
            dims=self.COUNTS_DIMS,
            coords={
                'age_group': self.age_group,
                'risk_group': self.risk_group,
                'compartment': self.compartment,
                'vertex': self.vertex
            }
        )

