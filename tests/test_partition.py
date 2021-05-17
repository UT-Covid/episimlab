from episimlab import apply_counts_delta
import logging
import pytest
import os
import pandas as pd
import numpy as np
import yaml
import xarray as xr
import xsimlab as xs
from itertools import product


from episimlab.partition.partition import Partition2Contact
from episimlab.models import basic
from episimlab.setup import epi


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

class TestPartitionInModel:

    def run_model(self, model, step_clock, input_vars, output_vars):
        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        # breakpoint()
        return input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))

    def test_partition_from_nc(self, step_clock):
        model = basic.partition()
        input_vars = dict(
            read_config__config_fp='tests/config/example_v2.yaml',
            get_contact_xr__contact_da_fp='tests/data/20200311_contact_matrix.nc',
        )
        output_vars = dict(apply_counts_delta__counts='step')
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)

    @pytest.mark.slow
    @pytest.mark.skipif(not os.path.isfile("data/20200311_travel.csv"),
                        reason="Very slow test that requires data/20200311_travel.csv")
    @pytest.mark.skip
    def test_partition_from_csv(self, step_clock):
        model = basic.partition().update_processes(dict(get_contact_xr=Partition2Contact))
        input_vars = dict(
            read_config__config_fp='tests/config/example_v2.yaml',
            get_contact_xr__travel_fp='data/20200311_travel.csv',
            get_contact_xr__contacts_fp='tests/data/polymod_contacts.csv',
        )
        output_vars = dict(apply_counts_delta__counts='step')
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)


class TestPartitioning:
    """
    Check that refactored partitioning generates expected results
    """

    def test_partitioning(self, updated_results, counts_coords_toy):
        inputs = {k: updated_results[k] for k in ('contacts_fp', 'travel_fp')}
        inputs.update({
            # 'age_group': counts_coords_toy['age_group'],
            # 'risk_group': counts_coords_toy['risk_group']
        })
        kw = dict(step_delta=np.timedelta64(24, 'h'),
                  step_start=np.datetime64('2020-03-11T00:00:00.000000000'),
                  step_end=np.datetime64('2020-03-12T00:00:00.000000000'),)
        proc = Partition2Contact(**inputs)
        proc.initialize(**kw)
        proc.run_step(**kw)

        tc_final = pd.read_csv(updated_results['tc_final_fp'], index_col=None)
        tr_part = pd.read_csv(updated_results['tr_parts_fp'], index_col=None)

        # test against legacy
        pd.testing.assert_frame_equal(proc.prob_partitions, tr_part.drop('Unnamed: 0', axis=1))
        pd.testing.assert_frame_equal(proc.contact_partitions, tc_final.drop('Unnamed: 0', axis=1))

    def test_phi(self, to_phi_da, updated_results, counts_coords_toy):
        inputs = {k: updated_results[k] for k in ('contacts_fp', 'travel_fp')}
        proc = Partition2Contact(**inputs)
        kw = dict(step_delta=np.timedelta64(24, 'h'),
                  step_start=np.datetime64('2020-03-11T00:00:00.000000000'),
                  step_end=np.datetime64('2020-03-12T00:00:00.000000000'),)
        proc.initialize(**kw)
        proc.run_step(**kw)

        # construct a DataArray from legacy phi
        phi = to_phi_da(updated_results['phi_fp'])

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da

        xr.testing.assert_allclose(sort_coords(proc.contact_xr), sort_coords(phi))
