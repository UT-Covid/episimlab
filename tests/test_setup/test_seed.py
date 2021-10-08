import pytest
import xsimlab as xs
import xarray as xr
import numpy as np
import logging
from numbers import Number

from episimlab.setup.seed import SeedGenerator


class TestSeedGenerator:

    @pytest.mark.parametrize('step, expected', [
        (1, 1457248422),
        (10, 3322450338),
        (100, 3734984104),
    ])
    def test_can_run_step(self, step, expected):
        inputs = {
            'seed_entropy': 12345
        }
        proc = SeedGenerator(**inputs)
        proc.initialize()

        for _ in range(step):
            proc.run_step()
        result = proc.seed_state

        assert isinstance(result, Number), type(result)
        assert result == expected
