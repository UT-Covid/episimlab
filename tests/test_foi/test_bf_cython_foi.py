import pytest
import logging
import xarray as xr
from episimlab.foi.bf_cython import BruteForceCythonFOI
from episimlab.foi.brute_force import BruteForceFOI


class TestBruteForceCythonFOI:

    def test_can_run_step(self, beta, omega, counts_basic, phi_t):
        """
        """
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            'vertex': counts_basic.coords['vertex'],
            'beta': beta,
            'omega': omega,
            'counts': counts_basic,
            'phi_t': phi_t,
        }
        foi_getter = BruteForceCythonFOI(**inputs)
        foi_getter.run_step()
        result = foi_getter.foi

        # assert that FOI is non zero
        assert result.sum() >= 1e-8
        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

    def test_can_reproduce_python(self, beta, omega, counts_basic, phi_t):
        """
        """
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            'vertex': counts_basic.coords['vertex'],
            'beta': beta,
            'omega': omega,
            'counts': counts_basic,
            'phi_t': phi_t,
        }

        # get FOI in cython
        cy_proc = BruteForceCythonFOI(**inputs)
        cy_proc.run_step()
        cy_result = cy_proc.foi
        assert isinstance(cy_result, xr.DataArray)

        # same but in python
        py_proc = BruteForceFOI(**inputs)
        py_proc.run_step()
        py_result = py_proc.foi
        assert isinstance(py_result, xr.DataArray)

        # assert that FOI is non zero
        assert cy_result.sum() >= 1e-8
        # assert equality
        xr.testing.assert_allclose(cy_result, py_result)
