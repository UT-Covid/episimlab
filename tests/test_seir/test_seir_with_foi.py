import pytest
import logging
import xarray as xr
from episimlab.seir.seir_with_foi import SEIRwithFOI


class TestSEIRwithFOI:

    def test_can_run_step(self, foi, seed_entropy, stochastic, counts_basic,
                          step_delta, epis, beta, omega, phi_t):
        """
        """
        inputs = {
            'counts': counts_basic,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
            'beta': beta,
            'omega': omega,
            'phi_t': phi_t,
        }
        inputs.update(epis)
        proc = SEIRwithFOI(**inputs)
        proc.run_step(step_delta)

        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

        # TODO
        # assert not result.isnull().any()
