import logging
import pytest
import os
import pandas as pd
import numpy as np
import yaml
import xarray as xr
import xsimlab as xs
from itertools import product


from episimlab.partition import toy, from_travel, partition
from episimlab.partition.travel_management import load_travel_df
from episimlab.models import basic
from episimlab.setup import epi


@pytest.fixture
def model():
    return basic.toy_partition()


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


@pytest.fixture(params=range(8))
def legacy_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'baseline_contact_df': load_travel_df(os.path.join(base_dir, f'contacts{idx}.csv')),
        'travel_df': load_travel_df(os.path.join(base_dir, f'travel{idx}.csv')),
        'tc_final_fp': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts_fp': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi_fp': os.path.join(base_dir, f'phi{idx}.npy'),
    }


@pytest.fixture(params=range(9))
def updated_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'baseline_contact_df': load_travel_df(os.path.join(base_dir, f'contacts{idx}.csv')),
        'travel_df': load_travel_df(os.path.join(base_dir, f'travel{idx}.csv')),
        'tc_final_fp': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts_fp': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi_fp': os.path.join(base_dir, f'phi{idx}.npy'),
    }

class TestToyPartitioning:
    """Can we migrate KP toy contact partitioning into episimlab processes?
    Do the migrated processes produce the same results as SEIR_Example?
    """

    def test_toy_partitioning(self, legacy_results_toy):
        inputs = {k: legacy_results_toy[k] for k in ('contacts_fp', 'travel_fp')}
        proc = toy.NaiveMigration(**inputs)
        proc.initialize()
        tc_final = pd.read_csv(legacy_results_toy['tc_final_fp'], index_col=None)
        phi = np.load(legacy_results_toy['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)
        np.testing.assert_array_almost_equal(proc.phi_ndarray, phi)

    def test_with_methods(self, to_phi_da, legacy_results_toy, counts_coords_toy,
                          phi_grp_mapping, step_delta):
        inputs = {k: legacy_results_toy[k] for k in ('contacts_fp', 'travel_fp')}
        inputs.update({
            'age_group': counts_coords_toy['age_group'],
            'risk_group': counts_coords_toy['risk_group'],
            'vertex': counts_coords_toy['vertex'],
            'phi_grp_mapping': phi_grp_mapping
        })
        proc = toy.SetupPhiWithToyPartitioning(**inputs)
        proc.initialize()
        proc.run_step(step_delta=step_delta)
        tc_final = pd.read_csv(legacy_results_toy['tc_final_fp'], index_col=None)

        # construct a DataArray from legacy phi
        phi = to_phi_da(legacy_results_toy['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da
        xr.testing.assert_allclose(sort_coords(proc.phi4d), sort_coords(phi))

        # ensure that phi4d and phi_t really have the same data
        pgm = proc.phi_grp_mapping
        pgm_coords = [proc.phi_grp_mapping.coords[k].values for k in pgm.dims]
        for v1, a1, r1, v2, a2, r2 in product(*pgm_coords * 2):
            pg1 = int(pgm.loc[{'vertex': v1, 'age_group': a1, 'risk_group': r1}])
            pg2 = int(pgm.loc[{'vertex': v2, 'age_group': a2, 'risk_group': r2}])
            res4d = proc.phi4d.loc[{
                'vertex1': v1,
                'vertex2': v2,
                'age_group1': a1,
                'age_group2': a2,
            }]
            res2d = proc.phi_t.loc[{'phi_grp1': pg1, 'phi_grp2': pg2}]
            assert res4d == res2d

    def test_consistent_with_xarray(self, to_phi_da, legacy_results_toy, step_delta,
                                    counts_coords_toy, phi_grp_mapping):
        """Is the xarray implementation consistent with the original one that uses
        pandas dataframes?
        """
        inputs = {k: legacy_results_toy[k] for k in ('contacts_fp', 'travel_fp')}
        inputs.update({
            'age_group': counts_coords_toy['age_group'],
            'risk_group': counts_coords_toy['risk_group'],
            'vertex': counts_coords_toy['vertex'],
            'phi_grp_mapping': phi_grp_mapping
        })

        # run reference process
        ref_proc = toy.SetupPhiWithToyPartitioning(**inputs)
        ref_proc.initialize()
        ref_proc.run_step(step_delta=step_delta)

        # run test process (the xarray implementation)
        test_proc = from_travel.SetupPhiWithPartitioning(**inputs)
        test_proc.initialize()
        test_proc.run_step(step_delta=step_delta)

        # assert equality
        xr.testing.assert_allclose(ref_proc.phi_t, test_proc.phi_t)


class TestSixteenComptToy:
    """Can we take the toy partitioning pipeline tested above, and have
    it pass phi parameter to the 'production' 16-compartment models
    in episimlab?

    TODO: recapitulate KP model sanity checks (partitioning contacts in
    different ways produces same result)
    """

    def run_model(self, model, step_clock, input_vars, output_vars):
        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        return input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))

    @pytest.mark.parametrize('config_fp', [
        'tests/data/partition_capture/test_config0.yaml'
    ])
    def test_can_run_model(self, epis, model, counts_basic,
                           step_clock, config_fp):
        # TODO: update step clock from config
        input_vars = {
            'read_config__config_fp': config_fp,
            'rng__seed_entropy': 12345,
            'sto__sto_toggle': -1,
            'setup_coords__n_age': 2,
            'setup_coords__n_nodes': 2,
            'setup_coords__n_risk': 1,
            'setup_phi__travel_fp': './tests/data/partition_capture/travel0.csv',
            'setup_phi__contacts_fp': './tests/data/partition_capture/contacts0.csv',
        }
        output_vars = {'apply_counts_delta__counts': 'step'}
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)


class TestPartitioning:
    """
    Check that refactored partitioning generates expected results
    """

    @pytest.mark.xfail(reason="Legacy dataframe missing some rows expected to contain zero contacts.")
    def test_partitioning(self, updated_results, counts_coords_toy):
        inputs = {k: updated_results[k] for k in ('baseline_contact_df', 'travel_df')}
        inputs.update({
            'age_group': counts_coords_toy['age_group'],
            'risk_group': counts_coords_toy['risk_group']
        })
        proc = partition.Partition(**inputs)
        proc.initialize()

        tc_final = pd.read_csv(updated_results['tc_final_fp'], index_col=None)

        # test against legacy
        pd.testing.assert_frame_equal(proc.contact_partitions, tc_final)

    def test_phi(self, to_phi_da, updated_results, counts_coords_toy):
        inputs = {k: updated_results[k] for k in ('baseline_contact_df', 'travel_df')}
        inputs.update({
            'age_group': counts_coords_toy['age_group'],
        })
        proc = partition.Partition(**inputs)
        proc.initialize()

        # construct a DataArray from legacy phi
        phi = to_phi_da(updated_results['phi_fp'])

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da

        # rename test output to have legacy coordinate names
        test_phi_xr = proc.contact_xr.rename(
            {'age_i': 'age_group1', 'age_j': 'age_group2', 'vertex_i': 'vertex1', 'vertex_j': 'vertex2'}
        )
        xr.testing.assert_allclose(sort_coords(test_phi_xr), sort_coords(phi))
