import pytest
import os


@pytest.fixture(params=range(14))
def legacy_results(request):
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    idx = request.param
    return {
        'contacts': os.path.join(base_dir, f'contacts{idx}.csv'),
        'travel': os.path.join(base_dir, f'travel{idx}.csv'),
        'tc_final': os.path.join(base_dir, f'tc_final{idx}.csv'),
        'tr_parts': os.path.join(base_dir, f'tr_parts{idx}.csv'),
        'phi': os.path.join(base_dir, f'phi{idx}.arr'),
    }


def test_toy_partitioning(legacy_results):
    print(legacy_results)
    assert 0
