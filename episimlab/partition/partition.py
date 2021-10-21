import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
import dask.dataframe as dd
from datetime import datetime
from ..utils import group_dict_by_var, get_int_per_day, fix_coord_dtypes, get_var_dims
from ..foi import BaseFOI


@xs.process
class Partition:
    TRAVEL_PAT_DIMS = (
        # formerly age, source, destination
        'vertex0', 'vertex1', 'age0', 
    )
    CONTACTS_DIMS = (
        'age0', 'age1', 
    )

    phi = xs.global_ref('phi', intent='out')
    travel_pat = xs.variable(
        dims=TRAVEL_PAT_DIMS, intent='in', global_name='travel_pat', 
        description="mobility/travel patterns")
    contacts = xs.variable(
        dims=CONTACTS_DIMS, intent='in', global_name='contacts', 
        description="pairwise baseline contact patterns")
    _coords = xs.group_dict('coords')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)

    def unsuffixed_coords(self, dims):
        return {d: self.coords.get(d.rstrip('01')) for d in dims}

    @property
    def phi_dims(self):
        """Overwrite this method if using different dims than `BaseFOI.PHI_DIMS`
        """
        return get_var_dims(BaseFOI, 'phi')

    @property
    def travel_pat_dims(self):
        return self.TRAVEL_PAT_DIMS

    @property
    def contacts_dims(self):
        return self.CONTACTS_DIMS

    @property
    def phi_coords(self):
        return self.unsuffixed_coords(self.phi_dims)

    @property
    def travel_pat_coords(self):
        return self.unsuffixed_coords(self.phi_dims)

    @property
    def contacts_coords(self):
        return self.unsuffixed_coords(self.contacts_dims)
    
    @xs.runtime(args=('step_delta',))
    def run_step(self, step_delta):
        """
        """
        int_per_day = get_int_per_day(step_delta)
        self.c_ijk = self.get_c_ijk(self.travel_pat)
        self.phi = self.c_ijk * self.contacts / int_per_day
    
    def get_c_ijk(self, tp: xr.DataArray) -> xr.DataArray:
        """
        """
        tp = tp.rename({'vertex1': 'k'}) # AKA 'destination'
        # similar to {'vertex0': 'vertex1', 'age0': 'age1'}
        zero_to_one = {
            k: k.replace('0', '1') for k in self.travel_pat_dims if '0' in k
        }
        
        # Calculate probability of contact between i and j
        n_ik = tp
        n_i = n_ik.sum('k')
        n_jk = tp.rename(zero_to_one)
        n_k = n_jk.sum('vertex1')
        c_ijk = (n_ik / n_i) * (n_jk / n_k)

        # Final transforms, sums, munging
        expected_dims = [dim for dim in self.phi_dims if dim in c_ijk.dims]
        c_ijk = (c_ijk 
                 .fillna(0.)
                 .sum('k') 
                 .transpose(*expected_dims))
        return c_ijk


@xs.process
class TravelPatFromCSV:
    RAISE_NULL = False
    travel_pat_fp = xs.variable(static=True, intent='in', description="path to "
                                "a CSV file containing travel patterns")
    travel_pat = xs.global_ref('travel_pat', intent='out')

    def initialize(self):
        """
        """
        self.run_step(None, None, is_init=True)

    @xs.runtime(args=('step_start', 'step_end',))
    def run_step(self, step_start, step_end, is_init=False):
        """
        """
        df = self.get_travel_df()
        if is_init:
            df = df[df['date'] == df['date'].min()]
        else:
            df = df[self.get_date_mask(df['date'], step_start, step_end)]
        da = self.get_travel_da(df, chunks=None)
        
        # Validation
        assert not df.empty, f'No travel data between {step_start} and {step_end}'
        print(f'The date in Partition.get_travel_df is {df["date"].unique()}')

        # Handle null values
        if da.isnull().any():
            logging.error(f"{(100 * int(da.isnull().sum()) / da.size):.1f}% values "
                          "in travel DataFrame are null")
            if self.RAISE_NULL or da.isnull().all():
                raise ValueError("found null values in travel DataFrame:\n" 
                                 f"{da.where(da.isnull(), drop=True)}")
            else:
                da = da.fillna(0.)
        
        # Change coordinate dtypes from 'object' to unicode
        self.travel_pat = fix_coord_dtypes(da)
        
    def get_date_mask(self, date: pd.Series, step_start, step_end) -> pd.Series:
        """Given timestamps `step_start` and `step_end`, returns a mask
        for the travel dataframe. Special handling for NaT and cases where 
        `step_start` equals `step_end`.
        """
        isnull = (pd.isnull(step_start), pd.isnull(step_end))
        assert not all(isnull), f"both of `step_start` and `step_end` are null (NaT)"

        if isnull[0]:
            mask = (date == step_end)
        elif isnull[1]:
            mask = (date == step_start)
        elif step_start == step_end:
            mask = (date == step_start)
        else:
            assert step_start <= step_end
            mask = (date >= step_start) & (date < step_end)
        return mask
    
    def get_travel_df(self) -> pd.DataFrame:
        """Load travel patterns from a CSV file and run preprocessing."""
        df = pd.read_csv(self.travel_pat_fp)
        df['date'] = pd.to_datetime(df['date'])
        if 'age_src' in df.columns:
            df = df.rename(columns=dict(age_src='age'))
        return df
            
    def get_travel_da(self, df: pd.DataFrame, chunks: int = None) -> xr.DataArray:
        """Convert a DataFrame into a single DataArray, using Dask to chunk
        into `chunk` divisions if `chunk` is not None.
        """
        df = df[['source', 'destination', 'age', 'n']]
        df = df.set_index(['source', 'destination', 'age'])
        ds = xr.Dataset.from_dataframe(df)
        ds = ds.rename({'destination': 'vertex1', 'source': 'vertex0', 'age': 'age0', })
        if chunks:
            ds = ds.chunk(chunks=chunks)
        return ds['n']


@xs.process
class ContactsFromCSV:
    contacts_fp = xs.variable(intent='in')
    contacts = xs.global_ref('contacts', intent='out')

    def initialize(self):
        self.contacts = self.get_contacts_da(df=self.get_contacts_df())
    
    def get_contacts_df(self):
        return pd.read_csv(self.contacts_fp).rename(columns={
            'age1': 'age0',
            'age2': 'age1',
        })

    def get_contacts_da(self, df: pd.DataFrame) -> xr.DataArray:
        df = df[['age0', 'age1', 'daily_per_capita_contacts']]
        df = df.set_index(['age0', 'age1'])
        ds = xr.Dataset.from_dataframe(df)
        da = ds['daily_per_capita_contacts']
        # Change coordinate dtypes from 'object' to unicode
        return fix_coord_dtypes(da)
    