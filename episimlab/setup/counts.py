import logging
import pandas as pd
import xsimlab as xs
import xarray as xr

from ..compt_model import ComptModel
from .coords import SetupToyCoords
from ..utils import get_var_dims, group_dict_by_var

logging.basicConfig(level=logging.DEBUG)


@xs.process
class SetupToyState:
    state = xs.global_ref('state', intent='out')
    _coords = xs.group_dict('coords')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)

    @property
    def state_dims(self):
        return get_var_dims(ComptModel, 'state')
    
    @property
    def state_coords(self):
        return {dim: self.coords[dim] for dim in self.state_dims}

    def initialize(self):
        self.state = self.realistic_state_basic()

    def uniform_state_basic(self, value=0.):
        return xr.DataArray(
            data=value,
            dims=self.counts_dims,
            coords=self.counts_coords
        )

    def realistic_state_basic(self):
        da = xr.DataArray(
            data=0.,
            dims=self.counts_dims,
            coords=self.counts_coords
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
class SetupStateFromCensusCSV(SetupToyState):
    """Initializes state from a census.gov formatted CSV file.
    """
    census_counts_csv = xs.variable(intent='in')
    
    def initialize(self):
        da = xr.DataArray(
            data=0.,
            dims=self.state_dims,
            coords=self.state_coords
        )
        dac = (self
               .read_census_csv()
               .reindex(dict(
                   vertex=self.coords['vertex'], 
                   age_group=self.coords['age_group']))
               # .expand_dims(['compartment', 'risk_group'])
               # .expand_dims(['risk_group'])
               # .transpose('vertex', 'age_group', ...)
               )
        # sanity checks
        assert not dac.isnull().any()
        assert all(zcta in dac.coords['vertex'] for zcta in da.coords['vertex'].values)

        # breakpoint()
        da.loc[dict(compartment='S', risk_group='low')] = dac
        self.state = da
        self.set_ia()

        # warning if detects no infected
        if self.state.loc[dict(compartment='Ia')].sum() < 1.:
            logging.warning(f"Population of Ia compartment is less than 1. Did " +
                          "you forget to set infected compartment?")

    def set_ia(self):
        """Sets Ia compartment to 50 for all vertices.
        """
        self.state.loc[dict(compartment='Ia', risk_group='low')] = 50.

    def read_census_csv(self) -> xr.DataArray:
        df = pd.read_csv(self.census_counts_csv)
        assert not df.isna().any().any(), ('found null values in df', df.isna().any())
        df.rename(columns={'GEOID': 'vertex', 'age_bin': 'age_group'}, inplace=True)
        df.set_index(['vertex', 'age_group'], inplace=True)
        # filter to zcta that we want to model in the simulation (vertex coords)
        df = df.loc[self.coords['vertex']]
        da = xr.DataArray.from_series(df['group_pop'])
        da.coords['age_group'] = da.coords['age_group'].astype(str)
        return da
