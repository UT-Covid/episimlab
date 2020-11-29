import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup.adj import InitToyAdj, InitAdjGrpMapping
from episimlab.pytest_utils import profiler


class TestInitAdj:

    # @profiler(log_dir='./logs')
    def test_can_run_step(self, counts_coords):
        inputs = {
            k: counts_coords[k] for k in
            ('age_group', 'risk_group', 'vertex', 'compartment')
        }
        # inputs['day_of_week'] = np.arange(7)
        proc = InitToyAdj(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        result = proc.adj
        assert isinstance(result, xr.DataArray)


class TestInitAdjGrpMapping:

    def test_can_init(self, counts_coords):
        inputs = {
            k: counts_coords[k] for k in
            ('age_group', 'risk_group', 'vertex', 'compartment')
        }
        proc = InitAdjGrpMapping(**inputs)
        proc.initialize()
        result = proc.adj_grp_mapping
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")
