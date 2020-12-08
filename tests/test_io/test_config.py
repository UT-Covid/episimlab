import pytest
import xsimlab as xs
import xarray as xr

from episimlab.io import config


@pytest.fixture
def coords_grp_dict(counts_coords):
    """Returns coordinate indices in a format that mimics xsimlab.group_dict"""
    return {('process_name', k): v for k, v in counts_coords}


class TestReadV1Config:

    def test_can_read_yaml(self, config_fp, counts_coords):
        inputs = dict(
            config_fp=config_fp,
            age_group=counts_coords['age_group'],
            risk_group=counts_coords['risk_group'],
        )
        proc = config.ReadV1Config(**inputs)
        result_dict = proc.get_config()
        assert isinstance(result_dict, dict), type(result_dict)

    def test_can_initialize(self, config_fp, counts_coords):
        inputs = dict(
            config_fp=config_fp,
            age_group=counts_coords['age_group'],
            risk_group=counts_coords['risk_group'],
        )
        proc = config.ReadV1Config(**inputs)
        proc.initialize()
