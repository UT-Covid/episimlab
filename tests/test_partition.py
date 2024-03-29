import logging
import pytest
import os
import pandas as pd
import numpy as np
import yaml
import xarray as xr
import xsimlab as xs
from itertools import product
from episimlab.partition import (
    Partition as PartitionUsingXR,
    ContactsFromCSV,
    TravelPatFromCSV,
)


@pytest.fixture
def step_delta():
    return np.timedelta64(144, 'm')


@pytest.fixture
def to_phi_da():
    def func(phi_fp):
        nodes = ['A', 'B', 'C']
        ages = ['young', 'old']
        data = np.load(phi_fp)
        shape = data.shape
        coords = {
            'vertex1': nodes[:shape[0]],
            'vertex2': nodes[:shape[1]],
            'age_group1': ages[:shape[2]],
            'age_group2': ages[:shape[3]]
        }
        phi = xr.DataArray(
            data=data,
            dims=('vertex1', 'vertex2', 'age_group1', 'age_group2'),
            coords=coords
        )
        return phi
    return func


@pytest.fixture
def phi_grp_mapping(counts_coords_toy):
    dims = ['vertex', 'age_group', 'risk_group']
    c = {k: v for k, v in counts_coords_toy.items() if k in dims}
    shape = [len(c[dim]) for dim in dims]
    data = range(np.product(shape))
    arr = np.array(data).reshape(shape)
    da = xr.DataArray(data=arr, dims=list(c.keys()), coords=c)
    return da


@pytest.fixture
def counts_coords_toy():
    return {
        'vertex': ['A', 'B'],
        'age_group': ['young', 'old'],
        'risk_group': ['low', 'high'],
        'compartment': ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                        'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                        'Py2Iy', 'Iy2Ih', 'H2D']
    }


@pytest.fixture(params=[
    'tests/data/partition_capture/test_config0.yaml'
])
def legacy_config(request):
    with open(request.param, 'r') as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture(params=range(8))
def legacy_results_toy(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'contacts_fp': os.path.join(base_dir, f'contacts{idx}.csv'),
        'travel_fp': os.path.join(base_dir, f'travel{idx}.csv'),
        'tc_final_fp': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts_fp': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi_fp': os.path.join(base_dir, f'phi{idx}.npy'),
    }


@pytest.fixture(params=[8, 9])
def updated_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'contacts_fp': os.path.join(base_dir, f'contacts{idx}.csv'),
        'travel_fp': os.path.join(base_dir, f'travel{idx}.csv'),
        'tc_final_fp': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts_fp': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi_fp': os.path.join(base_dir, f'phi{idx}.npy'),
    }


class TestPartition:

    def test_travel_xr_same_as_dask(self, updated_results, to_phi_da):
        """Check that xarray travel partitioning is consistent with 
        Dask implementation stored in `Partition2Contact.old_pr_contact_ijk`.
        """

        # use xarray implementation in old process
        # for generating travel_pat and contacts arrays only
        inputs = {k: updated_results[k] for k in ('contacts_fp', 'travel_fp')}
        kw = dict(step_delta=np.timedelta64(24, 'h'),
                  step_start=np.datetime64('2020-03-11T00:00:00.000000000'),
                  step_end=np.datetime64('2020-03-12T00:00:00.000000000'),)

        # new 3 process pipeline
        proc_tp = TravelPatFromCSV(travel_pat_fp=updated_results['travel_fp'])
        proc_tp.initialize()
        proc_tp.run_step(step_start=kw['step_start'], step_end=kw['step_end'])
        proc_ct = ContactsFromCSV(contacts_fp=updated_results['contacts_fp'])
        proc_ct.initialize()
        proc_xr = PartitionUsingXR(
            travel_pat=proc_tp.travel_pat,
            contacts=proc_ct.contacts,
        )
        proc_xr.run_step(step_delta=kw['step_delta'])

        # Check that phi is consistent with saved array
        phi = to_phi_da(updated_results['phi_fp']).rename({
            # rename to accommodate legacy dimension names
            'vertex1': 'vertex0',
            'vertex2': 'vertex1',
            'age_group1': 'age0',
            'age_group2': 'age1',
        })

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da

        xr.testing.assert_allclose(
            # new way
            sort_coords(proc_xr.phi),
            # archived result
            sort_coords(phi.transpose(*proc_xr.phi.dims)), 
        )