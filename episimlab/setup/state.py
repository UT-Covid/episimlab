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
            dims=self.state_dims,
            coords=self.state_coords
        )

    def realistic_state_basic(self):
        da = xr.DataArray(
            data=0.,
            dims=self.state_dims,
            coords=self.state_coords
        )
        # Houston
        da[dict(vertex=0)].loc[dict(compt='S')] = 2.326e6 / 10.
        # Austin
        da[dict(vertex=1)].loc[dict(compt='S')] = 1e6 / 10.
        # Beaumont
        da[dict(vertex=2)].loc[dict(compt='S')] = 1.18e5 / 10.
        # Start with 50 infected asymp in Austin
        da[dict(vertex=1)].loc[dict(compt='Ia')] = 50.
        return da


@xs.process
class SetupStateFromCensusCSV(SetupToyState):
    """Initializes state from a census.gov formatted CSV file.
    """
    census_state_csv = xs.variable(intent='in')
    
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
                   age=self.coords['age']))
               # .expand_dims(['compt', 'risk'])
               # .expand_dims(['risk'])
               # .transpose('vertex', 'age', ...)
               )
        # sanity checks
        assert not dac.isnull().any()
        assert all(zcta in dac.coords['vertex'] for zcta in da.coords['vertex'].values)

        # breakpoint()
        da.loc[dict(compt='S', risk='low')] = dac
        self.state = da
        self.set_ia()

        # warning if detects no infected
        if self.state.loc[dict(compt='Ia')].sum() < 1.:
            logging.warning(f"Population of Ia compt is less than 1. Did " +
                          "you forget to set infected compt?")

    def set_ia(self):
        """Sets Ia compt to 50 for all vertices.
        """
        self.state.loc[dict(compt='Ia', risk='low')] = 50.

    def read_census_csv(self) -> xr.DataArray:
        df = pd.read_csv(self.census_state_csv)
        assert not df.isna().any().any(), ('found null values in df', df.isna().any())
        df.rename(columns={'GEOID': 'vertex', 'age_bin': 'age'}, inplace=True)
        df.set_index(['vertex', 'age'], inplace=True)
        # filter to zcta that we want to model in the simulation (vertex coords)
        df = df.loc[self.coords['vertex']]
        da = xr.DataArray.from_series(df['group_pop'])
        da.coords['age'] = da.coords['age'].astype(str)
        return da
