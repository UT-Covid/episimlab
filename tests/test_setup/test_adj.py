import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup import InitToyAdj, InitAdjGrpMapping
from episimlab.pytest_utils import dask_prof

class TestInitAdj:

    # @dask_prof(log_dir='./logs')
    def test_can_run_step(self, counts_coords, adj_grp_mapping):
        inputs = {
            'adj_grp1': np.arange(10),
            'adj_grp2': np.arange(10),
            'adj_grp_mapping': adj_grp_mapping,
            'day_of_week': np.arange(7)
        }
        inputs.update({
            k: counts_coords[k] for k in
            ('age_group', 'risk_group', 'vertex', 'compartment')
        })
        proc = InitToyAdj(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        # compute intensive (180 s)
        # proc._toy_finalize_step()
        result = proc.adj
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")
        # logging.debug(f"proc.adj_t: {proc.adj_t}")


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
