import yaml
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
class SetupPhiWithPartitioning(NaiveMigration):
    PHI_DIMS = ('phi_grp1', 'phi_grp2')

    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    vertex = xs.global_ref('vertex')

    phi_t = xs.variable(dims=PHI_DIMS, intent='out', global_name='phi_t')

    def initialize(self):
        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)
        daily_timesteps = 10

        # Call functions from SEIR_Example
        self.tc_final = self.partition_contacts(self.travel, self.contacts,
                                                daily_timesteps=daily_timesteps)
        self.phi = self.contact_matrix(self.tc_final)
        self.phi_t = self.phi
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
    START_POP_DIMS = ('vertex', 'age_group')

    start_S = xs.variable(dims=START_POP_DIMS, intent='in')
    start_E = xs.variable(dims=START_POP_DIMS, intent='in')
    start_I = xs.variable(dims=START_POP_DIMS, intent='in')
    start_R = xs.variable(dims=START_POP_DIMS, intent='in')

    def initialize(self):
        self.counts = self.uniform_counts_basic(value=0.)
        self.set_counts('start_S', 'S')
        self.set_counts('start_E', 'E')
        self.set_counts('start_I', 'Ia')
        self.set_counts('start_R', 'R')

    def set_counts(self, attr_name, compt_name):
        val = getattr(self, attr_name)
        # TODO: unhardcode risk group specification
        self.counts.loc[{"risk_group": 0, "compartment": compt_name}] = val


@xs.process
class InitCoords(coords.InitDefaultCoords):
    age_group = xs.index(groups=['coords'], dims='age_group', global_name='age_group')
    risk_group = xs.index(groups=['coords'], dims='risk_group', global_name='risk_group')
    compartment = xs.index(groups=['coords'], dims='compartment', global_name='compartment')
    vertex = xs.index(groups=['coords'], dims='vertex', global_name='vertex')

    n_age = xs.variable(dims=(), intent='in')
    n_nodes = xs.variable(dims=(), intent='in')
    n_risk = xs.variable(dims=(), intent='in')

    def initialize(self):
        self.age_group = range(self.n_age)
        self.risk_group = range(self.n_risk)
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        self.vertex = range(self.n_nodes)


@xs.process
class ReadToyPartitionConfig:
    KEYS_MAPPING = {
        'beta': foi.base.BaseFOI,
        'omega': foi.base.BaseFOI,
        'mu': seir.base.BaseSEIR,
        'sigma': seir.base.BaseSEIR,
        'start_S': SetupCounts,
        'start_E': SetupCounts,
        'start_I': SetupCounts,
        'start_R': SetupCounts,
    }
    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    vertex = xs.global_ref('vertex')
    compartment = xs.global_ref('compartment')

    config_fp = xs.variable(static=True, intent='in')
    beta = xs.foreign(KEYS_MAPPING['beta'], 'beta', intent='out')
    omega = xs.foreign(KEYS_MAPPING['omega'], 'omega', intent='out')
    mu = xs.foreign(KEYS_MAPPING['mu'], 'mu', intent='out')
    sigma = xs.foreign(KEYS_MAPPING['sigma'], 'sigma', intent='out')
    start_S = xs.foreign(KEYS_MAPPING['start_S'], 'start_S', intent='out')
    start_E = xs.foreign(KEYS_MAPPING['start_E'], 'start_E', intent='out')
    start_I = xs.foreign(KEYS_MAPPING['start_I'], 'start_I', intent='out')
    start_R = xs.foreign(KEYS_MAPPING['start_R'], 'start_R', intent='out')

    def initialize(self):
        config = self.get_config()
        for name in self.KEYS_MAPPING:
            value = config.get(name, None)
            assert value is not None, f"{name} is None"
            setattr(self, name, self.try_coerce_to_da(value=value, name=name))

    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def try_coerce_to_da(self, name, value):
        """Given a variable with `name`, and `value` set from a config file,
        retrieve the variable metadata and use it to coerce the `value` into
        an `xarray.DataArray` with the correct dimensions and coordinates.
        Returns `value` if variable is scalar (zero length dims attribute),
        DataArray otherwise.
        """
        # get dims
        dims = get_var_dims(self.KEYS_MAPPING[name], name)
        if not dims:
            return value
        # get coords
        coords = {dim: getattr(self, dim) for dim in dims if dim != 'value'}
        da = xr.DataArray(data=value, dims=dims, coords=coords)
        # da.loc[dict()] = value
        return da
