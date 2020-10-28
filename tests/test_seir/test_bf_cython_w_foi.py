import pytest
import logging
import xarray as xr
from episimlab.seir.bf_cython_w_foi import BruteForceCythonWFOI


class TestBruteForceCythonWFOI:

    def test_can_run_step(self, phi_t, counts_basic, epis, beta, omega):
        inputs = {
            'beta': beta,
            'omega': omega,
            'counts': counts_basic,
            'phi_t': phi_t,
        }
        inputs.update(epis)

        proc = BruteForceCythonWFOI(**inputs)
        proc.run_step()
        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)



