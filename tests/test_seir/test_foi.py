import pytest
import logging
import xarray as xr
from episimlab.seir import BruteForceFOI
from numbers import Number


class TestFOIBruteForce:

    def test_can_import(self, counts_basic, phi_midx_mapping, phi_t):
        """
        """
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            'beta': 0.035,
            'omega': xr.DataArray(
                data=[0.667, 1., 0.9, 1.3],
                # TODO: age dim
                dims='compartment',
                coords=dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])
            ),
            'counts': counts_basic,
            'phi_t': phi_t,
            'phi_grp_mapping': phi_midx_mapping
        }
        foi_getter = BruteForceFOI(**inputs)
        foi_getter.run_step()
        result = foi_getter.foi

        # TODO: add accompanying MultiIndex mapping with every 2D array
        # phi and adjacency matrix

        logging.debug(f"phi_midx_mapping: {phi_midx_mapping}")
        #
        logging.debug(f"result: {result}")
        assert isinstance(result, Number)


