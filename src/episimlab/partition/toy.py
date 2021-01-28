import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
from itertools import product

from ..setup.coords import InitDefaultCoords
from .implicit_node import (
    partition_contacts, contact_matrix, probabilistic_partition)


@xs.process
class NaiveMigration:

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')

    def initialize(self):
        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)
        daily_timesteps = 10

        # Call functions from SEIR_Example
        self.tc_final = partition_contacts(self.travel, self.contacts,
                                           daily_timesteps=daily_timesteps)
        self.phi = contact_matrix(self.tc_final)
        self.phi_ndarray = self.phi.values


@xs.process
class WithMethods(NaiveMigration):

    age_group = xs.foreign(InitDefaultCoords, 'age_group')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group')
    vertex = xs.foreign(InitDefaultCoords, 'vertex')

    def initialize(self):
        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)
        daily_timesteps = 10

        # Call functions from SEIR_Example
        self.tc_final = self.partition_contacts(self.travel, self.contacts,
                                                daily_timesteps=daily_timesteps)
        self.phi = self.contact_matrix(self.tc_final)
        self.phi_ndarray = self.phi.values

    def partition_contacts(self, travel, contacts, daily_timesteps):
        tr_partitions = probabilistic_partition(travel, daily_timesteps)
        tc = pd.merge(tr_partitions, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2'])
        tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
        tc['partitioned_per_capita_contacts'] = tc['pr_contact_ij'] * tc['interval_per_capita_contacts']
        recalc = tc.groupby(['age_i', 'age_j'])['partitioned_per_capita_contacts'].sum().reset_index()
        recalc = pd.merge(recalc, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2']).dropna()
        tc = tc.dropna()
        tc_final = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']]
        return tc_final

    def contact_matrix(self, contact_df):
        ages = np.unique(contact_df[['age_i', 'age_j']])
        nodes = np.unique(contact_df[['i', 'j']])

        coords = {
            'vertex1': nodes,
            'vertex2': nodes,
            'age_group1': ages,
            'age_group2': ages
        }
        new_da = xr.DataArray(
            data=0.,
            dims=('vertex1', 'vertex2', 'age_group1', 'age_group2'),
            coords=coords
        )

        for n1, a1, n2, a2 in product(*[nodes, ages] * 2):
            subset = contact_df[(contact_df['i'] == n1) \
                & (contact_df['j'] == n2) \
                & (contact_df['age_i'] == a1) \
                & (contact_df['age_j'] == a2)]
            if subset.empty:
                val = 0
            else:
                val = subset['partitioned_per_capita_contacts'].item()
            new_da.loc[{
                'vertex1': n1,
                'vertex2': n2,
                'age_group1': a1,
                'age_group2': a2,
            }] = val
        return new_da
