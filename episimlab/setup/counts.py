import logging
import pandas as pd
import xsimlab as xs
import xarray as xr
from .coords import SetupToyCoords

from ..utils import get_var_dims

logging.basicConfig(level=logging.DEBUG)


@xs.process
class SetupToyState:

    # COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')
    COUNTS_DIMS = ApplyCountsDelta.COUNTS_DIMS

    counts = xs.global_ref('counts', intent='out')
    _coords = xs.group_dict('coords')

    @property
    def coords(self):
        return get_var_dims(self._coords)

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
class SetupCountsFromCensusCSV(SetupDefaultCounts):
    """Initializes counts from a census.gov formatted CSV file.
    """
    census_counts_csv = xs.variable(intent='in')
    
    def initialize(self):
        self.COUNTS_COORDS = {dim: getattr(self, dim) for dim in self.COUNTS_DIMS}
        da = xr.DataArray(
            data=0.,
            dims=self.COUNTS_DIMS,
            coords=self.COUNTS_COORDS
        )
        dac = (self
               .read_census_csv()
               .reindex(dict(
                   vertex=self.COUNTS_COORDS['vertex'], 
                   age_group=self.COUNTS_COORDS['age_group']))
               # .expand_dims(['compartment', 'risk_group'])
               # .expand_dims(['risk_group'])
               # .transpose('vertex', 'age_group', ...)
               )
        # sanity checks
        assert not dac.isnull().any()
        assert all(zcta in dac.coords['vertex'] for zcta in da.coords['vertex'].values)

        # breakpoint()
        da.loc[dict(compartment='S', risk_group='low')] = dac
        self.counts = da
        self.set_ia()

        # warning if detects no infected
        if self.counts.loc[dict(compartment='Ia')].sum() < 1.:
            logging.warning(f"Population of Ia compartment is less than 1. Did " +
                          "you forget to set infected compartment?")

    def set_ia(self):
        """Sets Ia compartment to 50 for all vertices.
        """
        self.counts.loc[dict(compartment='Ia', risk_group='low')] = 50.

    def read_census_csv(self) -> xr.DataArray:
        df = pd.read_csv(self.census_counts_csv)
        assert not df.isna().any().any(), ('found null values in df', df.isna().any())
        df.rename(columns={'GEOID': 'vertex', 'age_bin': 'age_group'}, inplace=True)
        df.set_index(['vertex', 'age_group'], inplace=True)
        # filter to zcta that we want to model in the simulation (vertex coords)
        df = df.loc[self.COUNTS_COORDS['vertex']]
        da = xr.DataArray.from_series(df['group_pop'])
        da.coords['age_group'] = da.coords['age_group'].astype(str)
        return da
