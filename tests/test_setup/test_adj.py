import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup import InitToyAdj, InitAdjGrpMapping


class TestInitAdj:

    def test_can_run_step(self, counts_coords, adj_grp_mapping):
        inputs = {
            'adj_grp1': np.arange(10),
            'adj_grp2': np.arange(10),
            'adj_grp_mapping': adj_grp_mapping,
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'day_of_week': np.arange(7)
        }
        proc = InitToyAdj(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        proc._toy_finalize_step()
        result = proc.adj
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")


class TestInitAdjGrpMapping:

    def test_can_init(self, counts_coords):
        inputs = {
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
        }
        proc = InitAdjGrpMapping(**inputs)
        proc.initialize()
        result = proc.adj_grp_mapping
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")
