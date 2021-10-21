import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
from ..utils import group_dict_by_var, get_int_per_day, fix_coord_dtypes, get_var_dims


@xs.process
class TravelPatFromCSV:
    """Load travel patterns from a CSV file to a `travel_pat` DataArray."""
    TAGS = ('partition', 'dependency::dask')
    RAISE_NULL = False
    travel_pat_fp = xs.variable(static=True, intent='in', description="path to "
                                "a CSV file containing travel patterns")
    travel_pat = xs.global_ref('travel_pat', intent='out')
    dask_chunks = xs.variable(static=True, intent='in', default=None,
                              global_name="dask_chunks",
                              description="number of chunks in which to divide "
                              "the `travel_pat` DataArray using Dask. None or 0 "
                              "will skip Dask chunking.")

    def initialize(self):
        self.run_step(None, None)

    @xs.runtime(args=('step_start', 'step_end',))
    def run_step(self, step_start, step_end):
        df = self.get_travel_df()

        # Both step_start and step_end will be None for initialize
        if step_start is None and step_end is None:
            df = df[df['date'] == df['date'].min()]
        else:
            df = df[self.get_date_mask(df['date'], step_start, step_end)]

        # Validation
        if df.empty:
            raise ValueError(f'No travel data between {step_start} and {step_end}')
        logging.info(f'The date in Partition.get_travel_df is {df["date"].unique()}')

        self.travel_pat = self.get_travel_da(df)
        
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
        elif self.dask_chunks:
            ds = ds.chunk(chunks=self.dask_chunks)
        da = ds['n']
            
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
        da = fix_coord_dtypes(da)
        return da


@xs.process
class TravelPatRepeatDaily(TravelPatFromCSV):
    """Example process that sets `travel_pat` based on a travel patterns
    CSV with data for only one date. This effectively sets `travel_pat` to be
    the same for the entire simulation.
    """
    TAGS = ('example', 'partition', 'dependency::dask')

    @xs.runtime(args=('step_start', 'step_end',))
    def run_step(self, step_start, step_end):
        df = self.get_travel_df()

        # Check that the travel_df only has one date
        assert len(df['date'].unique()) == 1, f"unique dates: {df['date'].unique()}"
        only_date = df['date'].unique()[0]

        # Both step_start and step_end will be None for initialize
        if step_start is None and step_end is None:
            pass
        elif only_date > step_start:
            raise ValueError(
                f"travel data has a single date ({only_date}) that is after "
                f"step start {step_start}.")

        self.travel_pat = self.get_travel_da(df)
    