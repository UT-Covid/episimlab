import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup import InitPhi, InitMidxMapping


class TestInitPhi:

    def test_can_init(self, counts_coords, phi_midx_mapping):
        inputs = {
            'midx1': np.arange(10),
            'midx2': np.arange(10),
            'midx_mapping': phi_midx_mapping,
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'day_of_week': np.arange(7)
        }
        proc = InitPhi(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        proc.finalize_step()
        result = proc.phi
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")


class TestInitMidxMapping:

    def test_can_init(self):
        inputs = {
            # foo
        }
        proc = InitMidxMapping(**inputs)
        proc.initialize()
        result = proc.midx_mapping
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")
