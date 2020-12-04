import pytest
import xsimlab as xs
import xarray as xr

from episimlab.io import config


@pytest.fixture
def config_fp():
    return './tests/data/config/example_v1.yaml'


class TestReadV1Config:

    def test_can_read_yaml(self, config_fp):
        inputs = dict(config_fp=config_fp)
        proc = config.ReadV1Config(**inputs)
        result_dict = proc.get_config()
        assert isinstance(result_dict, dict), type(result_dict)

    def test_can_initialize(self, config_fp):
        inputs = dict(config_fp=config_fp)
        proc = config.ReadV1Config(**inputs)
        proc.initialize()
