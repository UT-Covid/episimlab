import pytest
import xsimlab as xs
import xarray as xr
import numpy as np
import logging
from numbers import Number

from episimlab.setup.seed import SeedEntropy, SeedFromRNG


class TestSeedEntropy:

    def test_can_init(self):
        proc = SeedEntropy()
        proc.initialize()
        result = proc.seed_entropy

        assert isinstance(result, Number)


class TestSeedFromRNG:

    @pytest.mark.parametrize('step, expected', [
        (1, 959183449),
        (10, 1693324600),
        (100, 685568896),
    ])
    def test_can_run_step(self, step, expected):
        inputs = {
            'seed_entropy': 12345
        }
        proc = SeedFromRNG(**inputs)
        proc.initialize()

        for _ in range(step):
            proc.run_step()
        result = proc.seed_state

        assert isinstance(result, Number), type(result)
        assert result == expected

