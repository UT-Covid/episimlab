import pytest
import logging
import xarray as xr
from episimlab.setup.coords import InitDefaultCoords


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


