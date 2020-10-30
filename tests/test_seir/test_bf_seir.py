import pytest
import logging
import xarray as xr
from episimlab.seir.brute_force import BruteForceSEIR


class TestCountsDeltaSEIR:

    def test_can_run_step(self, foi, counts_basic, epis):
        inputs = {
            # 'age_group': counts_basic.coords['age_group'],
            # 'risk_group': counts_basic.coords['risk_group'],
            # 'beta': beta,
            # 'omega': omega,
            'counts': counts_basic,
            'foi': foi,
        }
        inputs.update(epis)

        proc = BruteForceSEIR(**inputs)
        proc.run_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)



