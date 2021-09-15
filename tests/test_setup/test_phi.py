import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup.phi import SetupToyPhi


class TestSetupPhi:

    def test_can_run_step(self, coords):
        proc = SetupToyPhi(coords=coords)
        proc.initialize()
        proc.run_step(step=0)
        result = proc.phi
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")

