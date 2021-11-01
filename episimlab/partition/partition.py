import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
from datetime import datetime
from ..utils import group_dict_by_var, get_int_per_day, fix_coord_dtypes, get_var_dims
from ..foi import BaseFOI


@xs.process
class Partition:
    """Given travel patterns `travel_pat` and baseline contact rates `contacts`,
    estimate the pairwise contact probabilities `phi` from contact partitioning.
    """
    TAGS = ('partition', )
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

