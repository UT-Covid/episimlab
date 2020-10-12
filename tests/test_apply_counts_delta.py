import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.apply_counts_delta import ApplyCountsDelta


class TestApplyCountsDelta:

    def test_can_finalize_step(self, counts_basic):
        inputs = {
            'counts': counts_basic,
            'counts_delta': [counts_basic] * 4
        }
        proc = ApplyCountsDelta(**inputs)
        proc.finalize_step()
        result = proc.counts
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")

