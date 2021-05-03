import yaml
import string
import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
from itertools import product

from .toy import SetupPhiWithToyPartitioning


@xs.process
class SetupPhiWithPartitioning(SetupPhiWithToyPartitioning):
    """Uses xarray, not pandas
    """
    PHI_DIMS = ('phi_grp1', 'phi_grp2')

    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    vertex = xs.global_ref('vertex')

    phi_grp_mapping = xs.global_ref('phi_grp_mapping')
    phi_t = xs.variable(dims=PHI_DIMS, intent='out', global_name='phi_t')

    def _initialize(self):
        self.phi_grp1 = self.phi_grp2 = range(self.phi_grp_mapping.size)
        self.PHI_COORDS = {k: getattr(self, k) for k in self.PHI_DIMS}

        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)

    @xs.runtime(args='step_delta')
    def _run_step(self, step_delta):
        # Get interval per day
        self.int_per_day = get_int_per_day(step_delta)

        # Call functions from SEIR_Example
        self.tc_final = self.partition_contacts(self.travel, self.contacts,
                                                daily_timesteps=self.int_per_day)
        self.phi4d = self.contact_matrix(self.tc_final)
        self.phi_t = self.convert_to_phi_grp(self.phi4d)
        self.phi_ndarray = self.phi4d.values

    def _convert_to_phi_grp(self, phi):
        """Converts 4-D array `arr` used in SEIR_Example to the flattened 2-D
        phi_t array expected by episimlab.
        """
        # initialize 2-D array of zeros
        da = xr.DataArray(data=0., dims=self.PHI_DIMS, coords=self.PHI_COORDS)
        # iterate over every unique pair of vertex, age group, risk group
        for v1, a1, r1, v2, a2, r2 in product(*[self.vertex, self.age_group,
                                                self.risk_group] * 2):
            # get phi groups for each set of coords
            pg1 = self.phi_grp_mapping.loc[{
                'vertex': v1,
                'age_group': a1,
                'risk_group': r1,
            }]
            pg2 = self.phi_grp_mapping.loc[{
                'vertex': v2,
                'age_group': a2,
                'risk_group': r2,
            }]
            # assign value
            value = phi.loc[{
                'vertex1': v1,
                'vertex2': v2,
                'age_group1': a1,
                'age_group2': a2,
            }]
            da.loc[{
                'phi_grp1': int(pg1),
                'phi_grp2': int(pg2),
            }] = value
        return da

    def _partition_contacts(self, travel, contacts, daily_timesteps):
        tr_partitions = probabilistic_partition(travel, daily_timesteps)
        tc = pd.merge(tr_partitions, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2'])
        tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
        tc['partitioned_per_capita_contacts'] = tc['pr_contact_ij'] * tc['interval_per_capita_contacts']
        recalc = tc.groupby(['age_i', 'age_j'])['partitioned_per_capita_contacts'].sum().reset_index()
        recalc = pd.merge(recalc, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2']).dropna()
        tc = tc.dropna()
        tc_final = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']]
        return tc_final

    def _contact_matrix(self, contact_df):
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
