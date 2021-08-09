import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup.phi import SetupPhi


class TestSetupPhi:

    def test_can_run_step(self, counts_coords):
        inputs = {
            'vertex': counts_coords['vertex'],
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            # 'compartment': counts_coords['compartment'],
        }
        proc = SetupPhi(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        result = proc.phi
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")

