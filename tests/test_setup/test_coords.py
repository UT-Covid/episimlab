import pytest
import logging
import xarray as xr
from episimlab.setup.coords import InitDefaultCoords, InitCoordsFromConfig


@pytest.fixture
def config_fp():
    return 'tests/config/example_v2.yaml'


class TestInitDefaultCoords:

    def test_can_initialize(self, counts_basic, phi_grp_mapping, phi_t):
        """
        """
        inputs = dict()
        proc = InitDefaultCoords(**inputs)
        proc.initialize()
        # result = proc.foi

        # logging.debug(f"phi_grp_mapping: {phi_grp_mapping}")
        # logging.debug(f"result: {result}")
        # assert isinstance(result, Number)


class TestInitCoordsFromConfig:

    def test_can_initialize(self, config_fp):
        """
        """
        inputs = dict(config_fp=config_fp)
        proc = InitCoordsFromConfig(**inputs)
        proc.initialize()
