import pytest
import os
import pandas as pd
import numpy as np

from episimlab.partition import toy


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

    def test_toy_partitioning(self, legacy_results):
        inputs = {k: legacy_results[k] for k in ('contacts_fp', 'travel_fp')}
        proc = toy.NaiveMigration(**inputs)
        proc.initialize()
        tc_final = pd.read_csv(legacy_results['tc_final_fp'], index_col=None)
        phi = np.load(legacy_results['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)
        np.testing.assert_array_almost_equal(proc.phi_ndarray, phi)


    def test_with_methods(self, legacy_results, counts_coords):
        inputs = {k: legacy_results[k] for k in ('contacts_fp', 'travel_fp')}
        inputs.update({
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'vertex': counts_coords['vertex']
        })
        proc = toy.WithMethods(**inputs)
        proc.initialize()
        tc_final = pd.read_csv(legacy_results['tc_final_fp'], index_col=None)
        phi = np.load(legacy_results['phi_fp'])

        # test against legacy
        pd.testing.assert_frame_equal(proc.tc_final, tc_final)
        np.testing.assert_array_almost_equal(proc.phi_ndarray, phi)


@pytest.mark.parametrize('expected', (
    'foo'
))
def _can_setup_static(self, counts_coords, tri_h2d, stochastic,
                      seed_state, expected):
    inputs = counts_coords.copy()
    inputs.update({
        'tri_h2d': tri_h2d,
        'stochastic': stochastic,
        'seed_state': seed_state,
    })

    proc = SetupStaticMuFromHtoD(**inputs)
    proc.initialize()
    result = proc.mu
    assert result == expected
