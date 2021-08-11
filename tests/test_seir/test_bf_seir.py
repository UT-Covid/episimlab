import pytest
import logging
import xarray as xr
from episimlab.seir.brute_force import BruteForceSEIR


class TestBruteForceSEIR:

    @pytest.mark.parametrize('n_steps', [
        10
    ])
    def test_can_run_step(self, seed_entropy, stochastic, foi,
                          counts_basic, epis, n_steps, step_delta, census_compt):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)

        proc = BruteForceSEIR(**inputs)

        # Check that the net change in population is still 0 in the census compartments
        for _ in range(n_steps):
            proc.run_step(step_delta)
            census = proc.counts_delta_seir.sel(dict(compartment=census_compt))
            compt_sums = {compt: proc.counts_delta_seir.sel(dict(compartment=compt)).sum() for compt in census_compt}
            assert abs(census.sum()) <= 1e-8

        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)
