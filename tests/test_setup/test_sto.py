import pytest
import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from episimlab.setup.sto import InitStochasticFromToggle


class TestInitStochasticFromToggle:

    @pytest.mark.parametrize('sto_toggle, step, expected', [
        (0., 0, True),
        (0., 10, True),
        (5.5, 5, False),
        (10, 10, True),
        (-1, 1000, False),
    ])
    def test_can_switch(self, sto_toggle, step, expected):
        inputs = {
            'sto_toggle': sto_toggle,
        }

        proc = InitStochasticFromToggle(**inputs)
        proc.run_step(step)
        result = proc.stochastic

        assert isinstance(result, bool)
        assert result == expected
