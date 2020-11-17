import pytest
import logging
import xarray as xr
from episimlab.seir.brute_force import BruteForceSEIR


class TestBruteForceSEIR:

    @pytest.mark.parametrize('n_steps', [
        10
    ])
    def test_can_run_step(self, seed_entropy, stochastic, foi,
                          counts_basic, epis, n_steps):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)

        proc = BruteForceSEIR(**inputs)

        # Check that the net change in population is still 0
        for _ in range(n_steps):
            proc.run_step()
            assert abs(proc.counts_delta_seir.sum()) <= 1e-8

        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)
