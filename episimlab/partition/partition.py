import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
from itertools import product
import dask.dataframe as dd
from datetime import datetime
from ..utils import group_dict_by_var

logging.basicConfig(level=logging.DEBUG)

# todo: vectorize
def contact_probability(n_i, n_j, n_i_total, n_k_total):

    # contacts between members of source node within the destination node
    try:
        fraction_i_to_j = n_i / n_i_total
    except ZeroDivisionError:
        fraction_i_to_j = 0

    try:
        fraction_j_in_k = n_j / n_k_total
    except ZeroDivisionError:
        fraction_j_in_k = 0

    pr_ii_in_j = fraction_i_to_j * fraction_j_in_k

    return pr_ii_in_j

def legacy_mapping(col_type, table):

    new_col = {
        'source_age': {
            'travel': 'age_src',
            'contact': 'age1'
        },
        'dest_age': {
            'travel': 'age_dest',
            'contact': 'age2'
        },
        'destination_age': {
            'travel': 'age_dest',
            'contact': 'age2'
        },
        'source_dem': {
            'travel': 'source_dem',
            'contact': 'source_dem'
        },
        'dest_dem': {
            'travel': 'dest_dem',
            'contact': 'dest_dem'
        }
    }

    return new_col[col_type][table]


@xs.process
class Partition2Contact:
    DIMS = ('vertex0', 'vertex1', 'age0', 'age1',)
    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')
    contact_xr = xs.variable(static=False, dims=DIMS, intent='out', global_name='contact_xr')

    @xs.runtime(args=['step_delta', 'step_start', 'step_end'])
    def initialize(self, step_delta, step_start, step_end):

        self.baseline_contact_df = pd.read_csv(self.contacts_fp)
        self.travel_df_with_date = self.load_travel_df()
        self.spatial_dims = ['source', 'destination']       # enforce that these are the only spatial dimensions
        self.age_dims = ['source_age', 'destination_age']          # always make age relative to source, destination

        # we need contact_xr set during initialize, for setting coordinates
        # time interval is set first timestep in travel df 
        step_end = self.travel_df_with_date.date.min().to_datetime64()
        self.run_step(None, step_start=step_end, step_end=step_end)
        assert hasattr(self, 'contact_xr')
        
    def get_travel_df(self) -> pd.DataFrame:
        """Given timestamps `step_start` and `step_end`, returns attr
        `travel_df`, which is indexed from attr `travel_df_with_date`.
        Special handling for NaT and cases where `step_start` equals `step_end`.
        """
        date = self.travel_df_with_date['date']

        isnull = (pd.isnull(self.step_start), pd.isnull(self.step_end))
        assert not all(isnull), \
            f"both of `step_start` and `step_end` are null (NaT)"
        if isnull[0]:
            mask = (date == self.step_end)
        elif isnull[1]:
            mask = (date == self.step_start)
        elif self.step_start == self.step_end:
            mask = (date == self.step_start)
        else:
            assert self.step_start <= self.step_end
            mask = (date >= self.step_start) & (date < self.step_end)
            
        # Generate travel_df by indexing on `date`
        self.travel_df = self.travel_df_with_date[mask]
        # DEBUG
        # self.travel_df[(self.travel_df['destination_type'] == 'local')]
        assert not self.travel_df.empty, \
            f'No travel data for date between {self.step_start} and {self.step_end}'
        print(f'The date in Partition.get_travel_df is {self.travel_df["date"].unique()}')
        return self.travel_df

    # NOTE: step_start and step_end reversed due to xarray-simlab bug
    @xs.runtime(args=['step_delta', 'step_end', 'step_start'])
    def run_step(self, step_delta, step_start, step_end):
        """Runs at every time step in the context of xsimlab.Model."""
        # propagate step metadata to instance scope
        self.step_delta = step_delta
        self.step_start = step_start
        self.step_end = step_end
        logging.debug(f"step_start: {self.step_start}")
        logging.debug(f"step_end: {self.step_end}")

        self.contacts = self.setup_contacts()
        # {'source': [], 'destination': [], 'source_age': [], 'destination_age': [], 'pr_contact_src_dest': []}
        self.all_dims = self.spatial_dims + self.age_dims
        # ['source', 'destination', 'source_age', 'destination_age']

        # would add demographic dims here if we had any, still trying to think through how to make certain dimensions optional...
        self.non_spatial_dims = self.age_dims  
        # ['source_age', 'destination_age']

        # Indexing on date, generate travel_df from travel_df_with_date
        self.travel_df = self.get_travel_df()
        #        Unnamed: 0  Unnamed: 0.1  source  destination    age            n       date destination_type
        # 0           30555         30555   76511        76511     <5    35.053846 2020-03-11            local
        # 1           30556         30556   76511        76511  18-49   472.284615 2020-03-11            local
        # 2           30557         30557   76511        76511   5-17   150.015385 2020-03-11            local
        # 3           30558         30558   76511        76511  50-64   165.469231 2020-03-11            local
        # 4           30559         30559   76511        76511    65+   121.369231 2020-03-11            local

        # initialize empty class members to hold intermediate results generated during workflow
        prob_partitions = self.dask_partition(self.travel_df)
        #         source_i  source_j  age_i  age_j  pr_contact_ijk
        # 0          76511     76511  18-49  18-49        0.374233
        # 1          76511     76511  18-49   5-17        0.369462
        # 2          76511     76511  18-49  50-64        0.361386
        # 3          76511     76511  18-49    65+        0.360255
        # 4          76511     76511  18-49     <5        0.361437
        contact_partitions = self.partitions_to_contacts(
            prob_partitions, contact_df=self.baseline_contact_df, daily_timesteps=1)
        #             i      j  age_i  age_j  partitioned_per_capita_contacts
        # 0       76511  76511  18-49  18-49                         3.830758
        # 1       76511  76527  18-49  18-49                         0.029453
        # 2       76511  76530  18-49  18-49                         0.303387
        # 3       76511  76537  18-49  18-49                         0.472820
        # 4       76511  76574  18-49  18-49                         1.071847
        self.contact_xr = self.build_contact_xr(contact_partitions)
        # Coordinates:
        # * vertex0  (vertex0) int64 76511 76527 76530 76537 ... 78758 78759 78953 78957
        # * vertex1  (vertex1) int64 76511 76527 76530 76537 ... 78758 78759 78953 78957
        # * age0     (age0) <U5 '18-49' '5-17' '50-64' '65+' '<5'
        # * age1     (age1) <U5 '18-49' '5-17' '50-64' '65+' '<5'
        self.contact_partitions = contact_partitions
        self.prob_partitions = prob_partitions

        df = self.travel_df[['source', 'destination', 'age', 'n', 'destination_type']]
        df = df.set_index(['source', 'destination', 'age', 'destination_type'])
        ds = xr.Dataset.from_dataframe(df)
        ds = ds.rename({'destination_type': 'dt', 'destination': 'k', 'source': 'i', 'age': 'age_i', }) 

        self.pr_contact_ijk = self.get_pr_c_ijk(da=ds.n)
    
    def get_pr_c_ijk(self, da: xr.DataArray, raise_null=False) -> xr.DataArray:
        """
        TODO: handle multiple `date`s
        """
        
        # Handle null values
        if da.isnull().any():
            logging.error(f"{(100 * int(da.isnull().sum()) / da.size):.1f}% values "
                          "in travel DataFrame are null")
            if raise_null or da.isnull().all():
                raise ValueError("found null values in travel DataFrame:\n" 
                                f"{da.where(da.isnull(), drop=True)}")
            else:
                da = da.fillna(0.)
        
        # Calculate probability of contact between i and j
        n_ik = da
        n_i = n_ik.sum(['k', 'dt'])
        n_jk = da.rename({'i': 'j', 'age_i': 'age_j', })
        n_k = n_jk.sum(['j', 'dt'])
        c_ijk = (n_ik / n_i) * (n_jk / n_k)
        return c_ijk.fillna(0.)

    def load_travel_df(self):

        tdf = pd.read_csv(self.travel_fp)
        tdf['date'] = pd.to_datetime(tdf['date'])
        try:
            tdf = tdf.rename(columns={'age_src': 'age'})
        except KeyError:
            pass

        return tdf

    def setup_contacts(self):

        contact_dict = {}
        for sd in self.spatial_dims:
            contact_dict[sd] = []
        for ad in self.age_dims:
            contact_dict[ad] = []
        contact_dict['pr_contact_src_dest'] = []

        return contact_dict
    
    @property
    def age(self, recalc=False) -> list:
        """Gets age group from join on travel_df and baseline_contact_df"""
        if not hasattr(self, '_age') or recalc is True:
            assert hasattr(self, 'baseline_contact_df')
            assert hasattr(self, 'travel_df')
            ag_from_contacts = set(self.baseline_contact_df[['age1', 'age2']].values.ravel('K'))
            ag_from_travel = set(self.travel_df.age.unique())
            self._age = ag_from_contacts.union(ag_from_travel)
        return list(self._age)

    def population_totals(self, travel_df):

        # get population totals for all locations (contextual and local areas)
        total_pop = travel_df.groupby(['source', 'age'])['n'].sum().reset_index()
        total_contextual = \
        travel_df[travel_df['destination_type'] == 'contextual'].groupby(['destination', 'age'])[
            'n'].sum().reset_index()
        total_pop = total_pop.rename(columns={'source': 'location'})
        total_contextual = total_contextual.rename(columns={'destination': 'location'})
        total = pd.concat([total_pop, total_contextual])

        return total

    def daily_totals(self, travel_df):

        # get daily population by destination
        daily_pop = travel_df.groupby(['destination', 'age'])['n'].sum().reset_index()

        return daily_pop

    def dask_partition(self, travel_df):

        total = self.population_totals(travel_df)
        #      location    age       n
        # 0       76511  18-49  1253.0
        # 1       76511   5-17   398.0
        # 2       76511  50-64   439.0
        # 3       76511    65+   322.0
        # 4       76511     <5    93.0
        daily_pop = self.daily_totals(travel_df)
        #     destination    age            n
        # 0          76511  18-49   489.766327
        # 1          76511   5-17   156.740512
        # 2          76511  50-64   176.553667
        # 3          76511    65+   129.770088
        # 4          76511     <5    37.177764

        # many:many self join to find sources that share common destinations
        logging.debug('Starting dask merge at {}'.format(datetime.now()))
        travel_left = dd.from_pandas(travel_df, npartitions=10)
        travel_right = dd.from_pandas(travel_df[['source', 'destination', 'age', 'n']], npartitions=100)
        travel_full = dd.merge(
            travel_left.set_index('destination').compute(),
            travel_right.set_index('destination').compute(),
            left_index=True,
            right_index=True,
            suffixes=['_i', '_j']
        ).reset_index()
        # print("travel_full:")
        # print(travel_full.head())
        logging.debug('Finishing dask merge at {}'.format(datetime.now()))

        # subsequent joins in pandas on unindexed data frames:
        # dask does not support multi-indexes
        # we need to maintain different names for the columns that would be the multi-index

        # add population totals to expanded travel dataframe
        # first merge adds n_total_i, the total population of i disregarding travel
        logging.debug('Starting pandas merge 1 at {}'.format(datetime.now()))
        travel_totals = pd.merge(
            travel_full, total,
            left_on=['source_i', 'age_i'],
            right_on=['location', 'age'],
            how='left',
        ).drop('location', axis=1)

        logging.debug('Starting pandas merge 2 at {}'.format(datetime.now()))
        # second merge adds n_total_k, the daily net population of k accounting for travel in and out by age group
        travel_totals = pd.merge(
            travel_totals, daily_pop,
            left_on=['destination', 'age_j'],
            right_on=['destination', 'age'],
            how='left',
            suffixes=['_total_i', '_total_k']
        )

        logging.debug('Calculating contact probabilities on full dataframe starting at {}'.format(datetime.now()))
        # calculate probability of contact between i and j in location k
        travel_totals['nij/ni'] = travel_totals['n_i'] / travel_totals['n_total_i']
        travel_totals['njk/nk'] = travel_totals['n_j'] / travel_totals['n_total_k']
        travel_totals['pr_contact_ijk'] = travel_totals['nij/ni'] * travel_totals['njk/nk']

        # DEBUG
        tt = travel_totals[['destination_type', 'age_i', 'age_j', 'destination', 'source_i', 'source_j', 
                            'n_i', 'n_total_i', 'n_j', 'n_total_k', 'pr_contact_ijk']]
        tt = tt.set_index(['destination_type', 'destination', 'source_j', 'age_j', 'source_i', 'age_i', ])
        tt_da = xr.Dataset.from_dataframe(tt)
        self.old_pr_contact_ijk = tt_da['pr_contact_ijk']

        # sum over contextual locations
        total_prob = travel_totals.groupby(['source_i', 'source_j', 'age_i', 'age_j'])[
            'pr_contact_ijk'].sum().reset_index()
        return total_prob

    def pandas_partition(self):

        total = self.population_totals()
        daily_pop = self.daily_totals()

        # one:many self join to find sources that share common destinations
        travel_full = pd.merge(
            self.travel_df,
            self.travel_df[['source', 'destination', 'age', 'n']],
            on='destination',
            suffixes=['_i', '_j']
        )

        # add population totals to expanded travel dataframe
        # first merge adds n_total_i, the total population of i disregarding travel
        travel_totals = pd.merge(
            travel_full, total,
            left_on=['source_i', 'age_i'],
            right_on=['location', 'age'],
            how='left',
        ).drop('location', axis=1)
        # second merge adds n_total_k, the daily net population of k accounting for travel in and out by age group
        travel_totals = pd.merge(
            travel_totals, daily_pop,
            left_on=['destination', 'age_j'],
            right_on=['destination', 'age'],
            how='left',
            suffixes=['_total_i', '_total_k']
        )

        # calculate probability of contact between i and j in location k
        travel_totals['nij/ni'] = travel_totals['n_i'] / travel_totals['n_total_i']
        travel_totals['njk/nk'] = travel_totals['n_j'] / travel_totals['n_total_k']
        travel_totals['pr_contact_ijk'] = travel_totals['nij/ni'] * travel_totals['njk/nk']

        # sum over contextual locations
        total_prob = travel_totals.groupby(['source_i', 'source_j', 'age_i', 'age_j'])['pr_contact_ijk'].sum().reset_index()
        return total_prob

    # todo: surface "daily_timesteps" to user
    def partitions_to_contacts(self, prob_partitions, contact_df, daily_timesteps):

        tc = pd.merge(
            prob_partitions, contact_df, how='outer',
            left_on=['age_i', 'age_j'],
            right_on=[legacy_mapping(i, 'contact') for i in self.non_spatial_dims]
        )
        tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
        tc['partitioned_per_capita_contacts'] = tc['pr_contact_ijk'] * tc['interval_per_capita_contacts']

        tc = tc.dropna()
        tc = tc.rename(columns={'source': 'i', 'destination': 'j', 'source_i': 'i', 'source_j': 'j', 'source_age': 'age_i', 'destination_age': 'age_j'})
        tc = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']].drop_duplicates()

        return tc

    def build_contact_xr(self, contact_partitions):

        logging.debug('Building contact xarray at {}'.format(datetime.now()))
        indexed_contact_df = contact_partitions.set_index(['i', 'j', 'age_i', 'age_j'])
        contact_xarray = indexed_contact_df.to_xarray()
        contact_xarray = contact_xarray.to_array()
        contact_xarray = contact_xarray.squeeze().drop('variable')
        contact_xarray = contact_xarray.rename(
            {
                'i': 'vertex0',
                'j': 'vertex1',
                'age_i': 'age0',
                'age_j': 'age1',
            }
        )

        # convert coords from dtype object
        contact_xarray.coords['age0'] = contact_xarray.coords['age0'].astype(str).values
        contact_xarray.coords['age1'] = contact_xarray.coords['age1'].astype(str).values

        #contact_xarray = contact_xarray.reset_coords(names='partitioned_per_capita_contacts', drop=True)
        return contact_xarray


@xs.process
class Contact2Phi:
    """Given array `contact_xr`, coerces to `phi_t` array."""
    PHI_DIMS = ('vertex0', 'vertex1',
                'age0', 'age1',
                'risk0', 'risk1')

    contact_xr = xs.global_ref('contact_xr', intent='in')
    phi_t = xs.variable(dims=PHI_DIMS, intent='out', global_name='phi_t')
    _coords = xs.group_dict('coords')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)
    
    @property
    def phi_dims(self):
        return self.PHI_DIMS

    @property
    def phi_coords(self):
        return {k: self.coords.get(k.rstrip('01')) for k in self.phi_dims}

    def initialize(self):
        self.get_phi()

    def run_step(self):
        self.get_phi()

    def get_phi(self):
        self.phi_t = xr.DataArray(data=np.nan, dims=self.phi_dims, 
                                  coords=self.phi_coords)
        # broadcast into phi_array
        # TODO: refactor
        self.phi_t.loc[dict(risk0='low', risk1='low')] = self.contact_xr
        self.phi_t.loc[dict(risk0='low', risk1='high')] = self.contact_xr
        self.phi_t.loc[dict(risk0='high', risk1='high')] = self.contact_xr
        self.phi_t.loc[dict(risk0='high', risk1='low')] = self.contact_xr
        assert not self.phi_t.isnull().any()


@xs.process
class NC2Contact:
    """Reads DataArray from NetCDF file at `contact_da_fp`, and sets attr
    `contact_xr`.
    """
    DIMS = ('vertex0', 'vertex1', 'age0', 'age1',)
    contact_da_fp = xs.variable(intent='in')
    contact_xr = xs.variable(dims=DIMS, intent='out', global_name='contact_xr')

    def initialize(self):
        da = (xr
              .open_dataarray(self.contact_da_fp)
              .rename({
                  'vertex_i': 'vertex0',
                  'vertex_j': 'vertex1',
                  'age_i': 'age0',
                  'age_j': 'age1',
               })
             )
        da.coords['age0'] = da.coords['age0'].astype(str)
        da.coords['age1'] = da.coords['age1'].astype(str)
        assert isinstance(da, xr.DataArray)

        self.contact_xr = da