import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
from itertools import product
from ..setup.coords import InitDefaultCoords
from .. import utils

logging.basicConfig(level=logging.DEBUG)


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
class Partition:

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')
    age_group = xs.foreign(InitDefaultCoords, 'age_group')  # age_groups = xs.variable(intent='in', default={'0-4', '5-17', '18-49', '50-64', '65+'})
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group')
    #time = xs.foreign(InitDefaultCoords, 'time')
    demographic_groups = xs.variable(intent='in', default=None)

    def initialize(self):

        self.baseline_contact_df = pd.read_csv(self.contacts_fp)
        self.travel_df = self.load_travel_df()

        self.age_group = self.age_group
        self.demographic_groups = self.demographic_groups
        # trying to think forward to other extensions; demographic groups are a placeholder for now but not implemented
        if self.demographic_groups:
            raise NotImplementedError
        self.spatial_dims = ['source', 'destination']       # enforce that these are the only spatial dimensions
        self.age_dims = ['source_age', 'destination_age']          # always make age relative to source, destination
        self.contacts = self.setup_contacts()
        self.all_dims = self.spatial_dims + self.age_dims
        self.non_spatial_dims = self.age_dims  # would add demographic dims here if we had any, still trying to think through how to make certain dimensions optional...
    
    # docs at https://xarray-simlab.readthedocs.io/en/latest/_api_generated/xsimlab.runtime.html?highlight=runtime#xsimlab.runtime
    @xs.runtime(args=['step_delta', 'step_start', 'step_end'])
    def run_step(self, step_delta, step_start, step_end):
        # step_delta is the time since previous step
        # Example of how to use the `step_delta` to convert to interval per day
        self.int_per_day = utils.get_int_per_day(step_delta)

        # step_start and step_end are datetimes marking beginning and end of this step
        logging.debug(f"step_start: {step_start}")
        logging.debug(f"step_end: {step_end}")



        # initialize empty class members to hold intermediate results generated during workflow
        self.prob_partitions = self.probabilistic_partition()
        self.contact_partitions = self.partitions_to_contacts(daily_timesteps=10)
        self.contact_xr = self.contact_matrix()
        # breakpoint()

    def subset_date(self, time_step_date):

        travel_current_date = self.travel_df[self.travel_df['date'] == time_step_date]
        try:
            assert travel_current_date.empty == False
        except AssertionError as e:
            e.args += (('No travel data for date {}.'.format(time_step_date), ))
            raise

        return travel_current_date

    def load_travel_df(self):

        tdf = pd.read_csv(self.travel_fp)
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

    def probabilistic_partition(self):

        total_pop = self.travel_df.groupby(['source', 'age'])['n'].sum().to_dict()
        total_contextual_dest = self.travel_df[self.travel_df['destination_type'] == 'contextual'].groupby(['destination', 'age'])['n'].sum().to_dict()

        if len(set(total_pop.keys()).intersection(set(total_contextual_dest.keys()))) > 0:
            raise ValueError('Contextual nodes cannot also be source nodes.')
        if len(set(total_contextual_dest.keys()).intersection(set(total_pop.keys()))) > 0:
            raise ValueError('Contextual nodes cannot also be source nodes.')

        total_pop.update(total_contextual_dest)

        # resident populations (population less those who travel out)
        residents = self.travel_df[self.travel_df['source'] == self.travel_df['destination']]
        residents = residents.groupby(['source', 'destination', 'age'])['n'].sum().to_dict()

        mapping = self.travel_df.groupby(['destination']).aggregate(lambda tdf: tdf.unique().tolist()).reset_index()
        mapping['source'] = [set(i) for i in mapping['source']]
        mapping['destination_type'] = [i[0] if len(i) == 1 else i for i in mapping['destination_type']]
        implicit2source = mapping[mapping['destination_type'] == 'contextual'][['source', 'destination']].set_index('destination').to_dict(orient='index')

        # if it's local contact, or contact in contextual location within local pop only, it's straightforward
        for i, row in self.travel_df.iterrows():
            if row['destination_type'] == 'local':
                for destination_age in self.age_group:

                    self.contacts['source'].append(row['source'])
                    self.contacts['destination'].append(row['destination'])
                    self.contacts['source_age'].append(row['age'])
                    self.contacts['destination_age'].append(destination_age)

                    # if it's local within-node contact, the pr(contact) = n stay in node / n total in node
                    # (no need to multiply by another fraction)
                    if (row['source'] == row['destination']) and (destination_age == row['age']):
                        daily_pr = contact_probability(
                            n_i=row['n'],                                       # people who do not leave source locality
                            n_j=1,                                              # set to 1 to ignore destination population (destination = source)
                            n_i_total=total_pop[(row['source'], row['age'])],   # total population of source locality
                            n_k_total=1                                         # set to 1 to ignore destination population (destination = source)
                        )
                        self.contacts['pr_contact_src_dest'].append(daily_pr)
                    else:
                        daily_pr = contact_probability(
                            n_i=row['n'],                                                               # migratory population from source locality
                            n_j=residents[(row['destination'], row['destination'], destination_age)],   # residential (daily) population of destination locality
                            n_i_total=total_pop[(row['source'], row['age'])],                           # total population of source locality
                            n_k_total=total_pop[(row['destination'], destination_age)]                  # total population of destination locality
                        )
                        self.contacts['pr_contact_src_dest'].append(daily_pr)

            # partitioning contacts between two different nodes within a contextual node requires a bit more parsing
            elif row['destination_type'] == 'contextual':

                # get all of the sources that feed into the contextual location
                other_sources = implicit2source[row['destination']]['source']
                for j in other_sources:

                    for destination_age in self.age_group:

                        self.contacts['source'].append(row['source'])
                        self.contacts['destination'].append(j)
                        self.contacts['source_age'].append(row['age'])
                        self.contacts['destination_age'].append(destination_age)

                        # filter by source and destination
                        j_to_dest = self.travel_df[(self.travel_df['source'] == j) \
                                                   & (self.travel_df['destination'] == row['destination']) \
                                                   & (self.travel_df['age'] == destination_age)]['n'].item()

                        # after all the filtering, there should be only one item remaining
                        daily_pr = contact_probability(
                            n_i=row['n'],                                               # migratory population from source locality
                            n_j=j_to_dest,                                              # migrants from other locality j that share same contextual destination
                            n_i_total=total_pop[(row['source'], row['age'])],           # total population of source locality
                            n_k_total=total_pop[(row['destination'], destination_age)]  # total population of contextual destination
                        )
                        self.contacts['pr_contact_src_dest'].append(daily_pr)

        contact_df = pd.DataFrame.from_dict(self.contacts)
        contact_df = contact_df.groupby(self.all_dims)['pr_contact_src_dest'].sum().reset_index()

        return contact_df

    def partitions_to_contacts(self, daily_timesteps):

        tc = pd.merge(
            self.prob_partitions, self.baseline_contact_df, how='outer',
            left_on=self.non_spatial_dims,
            right_on=[legacy_mapping(i, 'contact') for i in self.non_spatial_dims]
        )
        tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
        tc['partitioned_per_capita_contacts'] = tc['pr_contact_src_dest'] * tc['interval_per_capita_contacts']

        tc = tc.dropna()
        tc = tc.rename(columns={'source': 'i', 'destination': 'j', 'source_age': 'age_i', 'destination_age': 'age_j'})
        tc = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']].drop_duplicates()

        return tc

    def contact_matrix(self):

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
class TemporalPartition(Partition):

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')
    age_group = xs.foreign(InitDefaultCoords, 'age_group')  # age_groups = xs.variable(intent='in', default={'0-4', '5-17', '18-49', '50-64', '65+'})
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group')
    # vertex = xs.foreign(InitDefaultCoords, 'vertex')
    demographic_groups = xs.variable(intent='in', default=None)

    # todo: apply the contact partitioning over time slices in travel_fp