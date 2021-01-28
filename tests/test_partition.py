import pytest
import os

from episimlab.partition import toy


@pytest.fixture(params=range(14))
def legacy_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'contacts_fp': os.path.join(base_dir, f'contacts{idx}.csv'),
        'travel_fp': os.path.join(base_dir, f'travel{idx}.csv'),
        'tc_final': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi': os.path.join(base_dir, f'phi{idx}.arr'),
    }


def test_toy_partitioning(legacy_results):
    print(legacy_results)
    print(toy)
    inputs = {k: legacy_results[k] for k in ('contacts_fp', 'travel_fp')}
    proc = toy.NaiveMigration(**inputs)
    proc.initialize()


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
