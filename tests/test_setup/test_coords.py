import pytest
import logging
import xarray as xr
from episimlab.setup.coords import InitDefaultCoords, InitCoordsFromConfig


@pytest.fixture
def config_fp():
    return 'tests/config/example_v2.yaml'


class TestInitDefaultCoords:

    def test_can_initialize(self):
        """
        """
        inputs = dict()
        proc = InitDefaultCoords(**inputs)
        proc.initialize()


class TestInitCoordsFromConfig:

    def test_can_initialize(self, config_fp):
        """
        """
        inputs = dict(config_fp=config_fp)
        proc = InitCoordsFromConfig(**inputs)
        proc.initialize()
        for dim in ('vertex', 'compartment', 'age_group', 'risk_group'):
            assert hasattr(proc, dim)
