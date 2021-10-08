import pytest
import numpy as np
import xsimlab as xs
import xarray as xr
from math import isclose
from episimlab.utils import dt64_to_day_of_week


class TestDatetimeUtils:

    @pytest.mark.parametrize('arg, expected', (
        (np.datetime64('2018-01-01'), 0),
        (np.datetime64('2020-11-25'), 2),
    ))
    def test_dt64_to_day_of_week(self, arg, expected):
        result = dt64_to_day_of_week(arg)
        assert result == expected
