import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
from ..utils import fix_coord_dtypes


@xs.process
class ContactsFromCSV:
    """Load baseline contact patterns from a CSV file to a `contacts` DataArray.
    """
    TAGS = ('partition', 'dependency::pandas')
    contacts_fp = xs.variable(intent='in', description='Path to CSV file from '
                              'which to load baseline contact patterns')
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
    