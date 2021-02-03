import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
from itertools import product

from ..setup.coords import InitDefaultCoords
from .implicit_node import (
    partition_contacts, contact_matrix, probabilistic_partition)
from ..utils import get_var_dims
from .. import (foi, seir)
from ..setup import (seed, counts, coords, sto)


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


@xs.process
class SetupCounts(counts.InitDefaultCounts):

    start_S = xs.variable(dims=('age_group'), intent='in')
    start_E = xs.variable(dims=('age_group'), intent='in')
    start_I = xs.variable(dims=('age_group'), intent='in')
    start_R = xs.variable(dims=('age_group'), intent='in')

    def initialize(self):
        self.counts = self.uniform_counts_basic(value=0.)
        self.set_counts('start_S', 'S')
        self.set_counts('start_E', 'E')
        self.set_counts('start_I', 'I')
        self.set_counts('start_R', 'R')

    def set_counts(self, attr_name, compt_name):
        val = getattr(self, attr_name)
        self.counts.loc[{"compartment": compt_name}] = val


@xs.process
class InitCoords(coords.InitDefaultCoords):
    age_group = xs.index(groups=['coords'], dims='age_group')
    risk_group = xs.index(groups=['coords'], dims='risk_group')
    compartment = xs.index(groups=['coords'], dims='compartment')
    vertex = xs.index(groups=['coords'], dims='vertex')

    n_age = xs.variable(dims=(), intent='in')
    n_nodes = xs.variable(dims=(), intent='in')

    def initialize(self):
        self.age_group = range(self.n_age)
        self.risk_group = [0]
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        self.vertex = range(self.n_nodes)


@xs.process
class ReadToyPartitionConfig:
    KEYS_MAPPING = {
        'seed_entropy': seed.SeedGenerator,
        'sto_toggle': sto.InitStochasticFromToggle,
        'beta': foi.base.BaseFOI,
        'omega': foi.base.BaseFOI,
        'mu': seir.base.BaseSEIR,
        'sigma': seir.base.BaseSEIR,
        'n_age': InitCoords,
        'n_nodes': InitCoords,
        'start_S': SetupCounts,
        'start_E': SetupCounts,
        'start_I': SetupCounts,
        'start_R': SetupCounts,
    }

    config_fp = xs.variable(static=True, intent='in')
    seed_entropy = xs.foreign(KEYS_MAPPING['seed_entropy'], 'seed_entropy',
                              intent='out')
    sto_toggle = xs.foreign(KEYS_MAPPING['sto_toggle'], 'sto_toggle',
                            intent='out')
    beta = xs.foreign(KEYS_MAPPING['beta'], 'beta', intent='out')
    omega = xs.foreign(KEYS_MAPPING['omega'], 'omega', intent='out')
    mu = xs.foreign(KEYS_MAPPING['mu'], 'mu', intent='out')
    sigma = xs.foreign(KEYS_MAPPING['sigma'], 'sigma', intent='out')
    n_age = xs.foreign(KEYS_MAPPING['n_age'], 'n_age', intent='out')
    n_nodes = xs.foreign(KEYS_MAPPING['n_nodes'], 'n_nodes', intent='out')
    start_S = xs.foreign(KEYS_MAPPING['start_S'], 'start_S', intent='out')
    start_E = xs.foreign(KEYS_MAPPING['start_E'], 'start_E', intent='out')
    start_I = xs.foreign(KEYS_MAPPING['start_I'], 'start_I', intent='out')
    start_R = xs.foreign(KEYS_MAPPING['start_R'], 'start_R', intent='out')

    def initialize(self):
        config = self.get_config()
        for name, value in config.items():
            setattr(self, name, self.try_coerce_to_da(value=value, name=name))

    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
