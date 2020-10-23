import pytest
import logging
import xarray as xr
from episimlab.seir.bf_cython_w_foi import BruteForceCythonWFOI


class TestBruteForceCythonWFOI:

    def test_can_run_step(self, counts_basic, epis, beta, omega):
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            # 'beta': beta,
            # 'omega': omega,
            'counts': counts_basic,
            # 'foi': 0.7,
        }
        inputs.update(epis)

        proc = BruteForceCythonWFOI(**inputs)
        proc.run_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)



