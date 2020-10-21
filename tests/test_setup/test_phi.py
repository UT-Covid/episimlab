import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup import InitPhi, InitPhiGrpMapping


class TestInitPhi:

    def test_can_run_step(self, counts_coords, phi_grp_mapping):
        inputs = {
            'phi_grp1': np.arange(10),
            'phi_grp2': np.arange(10),
            'phi_grp_mapping': phi_grp_mapping,
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'day_of_week': np.arange(7)
        }
        proc = InitPhi(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        proc._toy_finalize_step()
        result = proc.phi
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")


class TestInitPhiGrpMapping:

    def test_can_init(self, counts_coords):
        inputs = {
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
        }
        proc = InitPhiGrpMapping(**inputs)
        proc.initialize()
        result = proc.phi_grp_mapping
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")
