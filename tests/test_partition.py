import pytest
import os
import pandas as pd
import numpy as np
import yaml
import xarray as xr
import xsimlab as xs

from episimlab.partition import toy
from episimlab.models import basic
from episimlab.setup import epi


@pytest.fixture
def model():
    """"""
    starter_model = basic.cy_seir_cy_foi()
    model = starter_model.drop_processes([
        # "setup_coords",
        "setup_counts",
        "read_config",
        "setup_beta",
        "setup_eta",
        "setup_gamma",
        "setup_mu",
        "setup_nu",
        "setup_omega",
        "setup_pi",
        "setup_rho",
        "setup_sigma",
        "setup_tau",
    ])
    return model.update_processes({
        "setup_counts": toy.SetupCounts,
        "setup_coords": toy.InitCoords,
        "read_config": toy.ReadToyPartitionConfig,
        "setup_eta": epi.SetupDefaultEta,
        "setup_tau": epi.SetupDefaultTau,
        "setup_nu": epi.SetupDefaultNu,
        "setup_pi": epi.SetupDefaultPi,
        "setup_rho": epi.SetupDefaultRho,
        "setup_gamma": epi.SetupDefaultGamma,
    })


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
def counts_coords_simple(request):
    return {
        'vertex': ['A', 'B', 'C'],
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
def legacy_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'contacts_fp': os.path.join(base_dir, f'contacts{idx}.csv'),
        'travel_fp': os.path.join(base_dir, f'travel{idx}.csv'),
        'tc_final_fp': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts_fp': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi_fp': os.path.join(base_dir, f'phi{idx}.npy'),
    }


class TestToyPartitioning:
    """Can we migrate KP toy contact partitioning into episimlab processes?
    Do the migrated processes produce the same results as SEIR_Example?
    """

    def test_toy_partitioning(self, legacy_results):
        inputs = {k: legacy_results[k] for k in ('contacts_fp', 'travel_fp')}
        proc = toy.NaiveMigration(**inputs)
        proc.initialize()
        tc_final = pd.read_csv(legacy_results['tc_final_fp'], index_col=None)
        phi = np.load(legacy_results['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)
        np.testing.assert_array_almost_equal(proc.phi_ndarray, phi)

    def test_with_methods(self, to_phi_da, legacy_results, counts_coords_simple):
        inputs = {k: legacy_results[k] for k in ('contacts_fp', 'travel_fp')}
        inputs.update({
            'age_group': counts_coords_simple['age_group'],
            'risk_group': counts_coords_simple['risk_group'],
            'vertex': counts_coords_simple['vertex']
        })
        proc = toy.WithMethods(**inputs)
        proc.initialize()
        tc_final = pd.read_csv(legacy_results['tc_final_fp'], index_col=None)

        # construct a DataArray from legacy phi
        phi = to_phi_da(legacy_results['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da
        xr.testing.assert_allclose(sort_coords(proc.phi), sort_coords(phi))


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
        # TODO: replicate SEIR_Example config, e.g. via ReadToyPartitionConfig
        # TODO: update step clock from config
        input_vars = {
            'read_config__config_fp': config_fp,
            'rng__seed_entropy': 12345,
            'sto__sto_toggle': -1,
            'setup_coords__n_age': 2,
            'setup_coords__n_nodes': 2
        }
        output_vars = {'apply_counts_delta__counts': 'step'}
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)
