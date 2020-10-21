import pytest
import logging
import xarray as xr
from episimlab.graph.cython import CythonGraph
from numbers import Number


class TestCythonGraph:

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
        proc.run_step()

        logging.debug(f"proc.adj_t: {proc.adj_t.coords}")
        logging.debug(f"proc.adj_grp_mapping: {proc.adj_grp_mapping.coords}")
        # result = proc.foi

        # logging.debug(f"phi_grp_mapping: {phi_grp_mapping}")
        # logging.debug(f"result: {result}")
        # assert isinstance(result, Number)


