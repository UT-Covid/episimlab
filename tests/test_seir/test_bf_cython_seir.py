import pytest
import logging
import xarray as xr
from episimlab.seir.bf_cython import BruteForceCython
from episimlab.seir.brute_force import BruteForceSEIR


class TestBruteForceCython:

    def test_can_run_step(self, foi, seed_entropy, stochastic, counts_basic, epis):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)

        proc = BruteForceCython(**inputs)
        # proc.initialize()
        proc.run_step()
        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

        # TODO
        # assert not result.isnull().any()

    def test_same_as_python(self, foi, stochastic, seed_entropy, counts_basic, epis):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)

        # run in cython
        cy_proc = BruteForceCython(**inputs)
        # cy_proc.initialize()
        cy_proc.run_step()
        cy_proc.finalize_step()
        cy_result = cy_proc.counts_delta_seir
        assert isinstance(cy_result, xr.DataArray)

        # run in python
        py_proc = BruteForceSEIR(**inputs)
        # cy_proc.initialize()
        py_proc.run_step()
        # py_proc.finalize_step()
        py_result = py_proc.counts_delta_seir
        assert isinstance(py_result, xr.DataArray)

        # assert are the same
        xr.testing.assert_allclose(py_result, cy_result)
