import pytest
import logging
import xarray as xr
from episimlab.seir.bf_cython import BruteForceCython


class TestBruteForceCython:

    def test_can_run_step(self, foi, counts_basic, epis):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
        }
        inputs.update(epis)

        proc = BruteForceCython(**inputs)
        proc.run_step()
        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

