import pandas as pd
import xsimlab as xs
import xarray as xr
from .coords import InitDefaultCoords

from ..apply_counts_delta import ApplyCountsDelta



@xs.process
class InitDefaultCounts:

    # COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')
    COUNTS_DIMS = ApplyCountsDelta.COUNTS_DIMS

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='out')
    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    compartment = xs.global_ref('compartment')
    vertex = xs.global_ref('vertex')

    def initialize(self):
        self.counts = self.realistic_counts_basic()

    def uniform_counts_basic(self, value=0.):
        return xr.DataArray(
            data=value,
            dims=self.COUNTS_DIMS,
            coords={dim: getattr(self, dim) for dim in self.COUNTS_DIMS}
        )

    def realistic_counts_basic(self):
        da = xr.DataArray(
            data=0.,
            dims=self.COUNTS_DIMS,
            coords={dim: getattr(self, dim) for dim in self.COUNTS_DIMS}
        )
        # Houston
        da[dict(vertex=0)].loc[dict(compartment='S')] = 2.326e6 / 10.
        # Austin
        da[dict(vertex=1)].loc[dict(compartment='S')] = 1e6 / 10.
        # Beaumont
        da[dict(vertex=2)].loc[dict(compartment='S')] = 1.18e5 / 10.
        # Start with 50 infected asymp in Austin
        da[dict(vertex=1)].loc[dict(compartment='Ia')] = 50.

        return da

@xs.process
class InitCountsFromCensusCSV(InitDefaultCounts):
    """Initializes counts from a census.gov formatted CSV file.
    """
    
    def initialize(self):
        pass