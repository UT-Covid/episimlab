import pytest
import logging
import xarray as xr
from episimlab.graph.cython import CythonGraph
import numpy as np

from episimlab.pytest_utils import profiler


class TestCythonGraph:

    @profiler()
    def test_can_run_step(self, omega, counts_basic, counts_coords,
                          adj_grp_mapping, adj_t):
        """
        """
        inputs = {
            'counts': counts_basic,
            'adj_t': adj_t,
            'adj_grp_mapping': adj_grp_mapping
        }
        inputs.update({
            k: counts_coords[k] for k in
            ('age_group', 'risk_group', 'vertex', 'compartment')
        })
        proc = CythonGraph(**inputs)

        # logging.debug(f"proc.counts: {proc.counts.coords}")
        # logging.debug(f"proc.counts: {proc.counts.dims}")
        # logging.debug(f"proc.adj_grp_mapping: {proc.adj_grp_mapping.coords}")
        # logging.debug(f"proc.adj_grp_mapping.size: {proc.adj_grp_mapping.size}")
        logging.debug(f"proc.adj_grp_mapping.shape: {proc.adj_grp_mapping.shape}")

        proc.run_step()
        proc.finalize_step()
        result = proc.counts_delta_gph
        assert isinstance(result, xr.DataArray)

        # logging.debug(f"result.shape: {result.shape}")
        # logging.debug(f"result: {result}")

