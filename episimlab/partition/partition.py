import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
from itertools import product
from .. import utils
from ..pytest_utils import profiler
import dask.dataframe as dd
from datetime import datetime

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
    DIMS = ('vertex1', 'vertex2', 'age_group1', 'age_group2',)
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
        

    # docs at https://xarray-simlab.readthedocs.io/en/latest/_api_generated/xsimlab.runtime.html?highlight=runtime#xsimlab.runtime
    @xs.runtime(args=['step_delta', 'step_start', 'step_end'])
    def run_step(self, step_delta, step_start, step_end):

        # step_start and step_end are datetime64 marking beginning and end of this step
        logging.debug(f"step_start: {step_start}")
        logging.debug(f"step_end: {step_end}")
        # float('inf') for step_end if it is NaT

        if pd.isnull(step_end):
            step_end = pd.Timestamp.max

        # step_delta is the time since previous step
        # we could just as easily calculate this: step_end - step_start
        # Example of how to use the `step_delta` to convert to interval per day
        self.int_per_day = utils.get_int_per_day(step_delta) if step_delta else None

        self.contacts = self.setup_contacts()
        self.all_dims = self.spatial_dims + self.age_dims
        self.non_spatial_dims = self.age_dims  # would add demographic dims here if we had any, still trying to think through how to make certain dimensions optional...

        # Generate travel_df by indexing on `date`
        self.travel_df = self.travel_df_with_date[
            (self.travel_df_with_date['date'] >= step_start) &
            (self.travel_df_with_date['date'] <= step_end)
        ]
        assert not self.travel_df.empty, \
            f'No travel data for date between {step_start} and {step_end}'

        # initialize empty class members to hold intermediate results generated during workflow
        self.prob_partitions = self.dask_partition()
        self.contact_partitions = self.partitions_to_contacts(daily_timesteps=10)
        self.contact_xr = self.build_contact_xr()

    @profiler()
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
    def age_group(self, recalc=False) -> list:
        """Gets age group from join on travel_df and baseline_contact_df"""
        if not hasattr(self, '_age_group') or recalc is True:
            assert hasattr(self, 'baseline_contact_df')
            assert hasattr(self, 'travel_df')
            ag_from_contacts = set(self.baseline_contact_df[['age1', 'age2']].values.ravel('K'))
            ag_from_travel = set(self.travel_df.age.unique())
            self._age_group = ag_from_contacts.union(ag_from_travel)
        return list(self._age_group)

    def population_totals(self):

        # get population totals for all locations (contextual and local areas)
        total_pop = self.travel_df.groupby(['source', 'age'])['n'].sum().reset_index()
        total_contextual = \
        self.travel_df[self.travel_df['destination_type'] == 'contextual'].groupby(['destination', 'age'])[
            'n'].sum().reset_index()
        total_pop = total_pop.rename(columns={'source': 'location'})
        total_contextual = total_contextual.rename(columns={'destination': 'location'})
        total = pd.concat([total_pop, total_contextual])

        return total

    def daily_totals(self):

        # get daily population by destination
        daily_pop = self.travel_df.groupby(['destination', 'age'])['n'].sum().reset_index()

        return daily_pop

    def dask_partition(self):

        total = self.population_totals()
        daily_pop = self.daily_totals()

        # many:many self join to find sources that share common destinations
        print('Starting dask merge at {}'.format(datetime.now()))
        travel_left = dd.from_pandas(self.travel_df, npartitions=10)
        travel_right = dd.from_pandas(self.travel_df[['source', 'destination', 'age', 'n']], npartitions=100)
        travel_full = dd.merge(
            travel_left.set_index('destination').compute(),
            travel_right.set_index('destination').compute(),
            left_index=True,
            right_index=True,
            suffixes=['_i', '_j']
        ).reset_index()
        print('Finishing dask merge at {}'.format(datetime.now()))

        # subsequent joins in pandas on unindexed data frames:
        # dask does not support multi-indexes
        # we need to maintain different names for the columns that would be the multi-index

        # add population totals to expanded travel dataframe
        # first merge adds n_total_i, the total population of i disregarding travel
        print('Starting pandas merge 1 at {}'.format(datetime.now()))
        travel_totals = pd.merge(
            travel_full, total,
            left_on=['source_i', 'age_i'],
            right_on=['location', 'age'],
            how='left',
        ).drop('location', axis=1)

        print('Starting pandas merge 2 at {}'.format(datetime.now()))
        # second merge adds n_total_k, the daily net population of k accounting for travel in and out by age group
        travel_totals = pd.merge(
            travel_totals, daily_pop,
            left_on=['destination', 'age_j'],
            right_on=['destination', 'age'],
            how='left',
            suffixes=['_total_i', '_total_k']
        )

        print('Calculating contact probabilities on full dataframe starting at {}'.format(datetime.now()))
        # calculate probability of contact between i and j in location k
        travel_totals['nij/ni'] = travel_totals['n_i'] / travel_totals['n_total_i']
        travel_totals['njk/nk'] = travel_totals['n_j'] / travel_totals['n_total_k']
        travel_totals['pr_contact_ijk'] = travel_totals['nij/ni'] * travel_totals['njk/nk']

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
    def partitions_to_contacts(self, daily_timesteps):

        tc = pd.merge(
            self.prob_partitions, self.baseline_contact_df, how='outer',
            left_on=['age_i', 'age_j'],
            right_on=[legacy_mapping(i, 'contact') for i in self.non_spatial_dims]
        )
        tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
        tc['partitioned_per_capita_contacts'] = tc['pr_contact_ijk'] * tc['interval_per_capita_contacts']

        tc = tc.dropna()
        tc = tc.rename(columns={'source': 'i', 'destination': 'j', 'source_i': 'i', 'source_j': 'j', 'source_age': 'age_i', 'destination_age': 'age_j'})
        tc = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']].drop_duplicates()

        return tc

    def build_contact_xr(self):

        print('Building contact xarray at {}'.format(datetime.now()))
        indexed_contact_df = self.contact_partitions.set_index(['i', 'j', 'age_i', 'age_j'])
        contact_xarray = indexed_contact_df.to_xarray()
        contact_xarray = contact_xarray.to_array()
        contact_xarray = contact_xarray.squeeze().drop('variable')
        contact_xarray = contact_xarray.rename(
            {
                'i': 'vertex1',
                'j': 'vertex2',
                'age_i': 'age_group1',
                'age_j': 'age_group2',
            }
        )
        #contact_xarray = contact_xarray.reset_coords(names='partitioned_per_capita_contacts', drop=True)
        return contact_xarray

    def contact_matrix(self):

        print('Finalizing contact matrix at {}'.format(datetime.now()))
        sources = self.contact_partitions['i'].unique()
        destinations = self.contact_partitions['j'].unique()

        nodes = []
        for i in sources:
            nodes.append(i)
        for j in destinations:
            nodes.append(j)
        nodes = sorted(list(set(nodes)))

        arr_dims = []
        coords = {}

        # spatial dimensions and coordinates
        arr_dims.extend([len(nodes), len(nodes)])
        coords['vertex_i'] = nodes
        coords['vertex_j'] = nodes

        # age dimensions and coordinates
        if self.age_dims:
            arr_dims.extend([len(self.age_group), len(self.age_group)])  # age by source and destination
            coords['age_i'] = self.age_group
            coords['age_j'] = self.age_group

        new_da = xr.DataArray(
            data=0.,
            dims=(coords.keys()),
            coords=coords
        )

        # for now we are ignoring the possible demographic dimension
        for n1, a1, n2, a2 in product(*[nodes, self.age_group] * 2):
            subset = self.contact_partitions[(self.contact_partitions['i'] == n1) \
                & (self.contact_partitions['j'] == n2) \
                & (self.contact_partitions['age_i'] == a1) \
                & (self.contact_partitions['age_j'] == a2)]
            if subset.empty:
                val = 0
            else:
                val = subset['partitioned_per_capita_contacts'].item()
            new_da.loc[{
                'vertex_i': n1,
                'vertex_j': n2,
                'age_i': a1,
                'age_j': a2,
            }] = val

        return new_da


@xs.process
class Contact2Phi:
    """Given array `contact_xr`, coerces to `phi_t` array."""
    PHI_DIMS = ('vertex1', 'vertex2',
                'age_group1', 'age_group2',
                'risk_group1', 'risk_group2')

    age_group = xs.index(dims='age_group', global_name='age_group')
    risk_group = xs.index(dims='risk_group', global_name='risk_group')
    compartment = xs.index(dims='compartment', global_name='compartment')
    vertex = xs.index(dims='vertex', global_name='vertex')

    contact_xr = xs.global_ref('contact_xr', intent='in')
    phi_t = xs.variable(dims=PHI_DIMS, intent='out', global_name='phi_t')

    def initialize(self):
        self.initialize_misc_coords()
        self.run_step()

    def run_step(self):
        # set age group and vertex coords
        self.age_group = self.contact_xr.coords['age_group1'].values
        self.vertex = self.contact_xr.coords['vertex1'].values

        self.get_phi()

    def get_phi(self):
        self.COORDS = {k: getattr(self, k[:-1]) for k in self.PHI_DIMS}
        self.phi_t = xr.DataArray(data=np.nan, dims=self.PHI_DIMS, coords=self.COORDS)

        # broadcast into phi_array
        # TODO: refactor
        self.phi_t.loc[dict(risk_group1='low', risk_group2='low')] = self.contact_xr
        self.phi_t.loc[dict(risk_group1='low', risk_group2='high')] = self.contact_xr
        self.phi_t.loc[dict(risk_group1='high', risk_group2='high')] = self.contact_xr
        self.phi_t.loc[dict(risk_group1='high', risk_group2='low')] = self.contact_xr
        assert not self.phi_t.isnull().any()

    def initialize_misc_coords(self):
        """Set up coords besides vertex and age group."""
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']


@xs.process
class NC2Contact:
    """Reads DataArray from NetCDF file at `contact_da_fp`, and sets attr
    `contact_xr`.
    """
    DIMS = ('vertex1', 'vertex2', 'age_group1', 'age_group2',)
    contact_da_fp = xs.variable(intent='in')
    contact_xr = xs.variable(dims=DIMS, intent='out', global_name='contact_xr')

    @profiler()
    def initialize(self):
        da = (xr
              .open_dataarray(self.contact_da_fp)
              .rename({
                  'vertex_i': 'vertex1',
                  'vertex_j': 'vertex2',
                  'age_i': 'age_group1',
                  'age_j': 'age_group2',
               })
             )
        da.coords['age_group1'] = da.coords['age_group1'].astype(str)
        da.coords['age_group2'] = da.coords['age_group2'].astype(str)
        assert isinstance(da, xr.DataArray)

        self.contact_xr = da