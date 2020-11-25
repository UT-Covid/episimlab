import pytest
import numpy as np
import xsimlab as xs
import xarray as xr
from math import isclose

from episimlab.cy_utils.cy_utils import discrete_time_approx_wrapper as cy_dta
from episimlab.utils import discrete_time_approx as py_dta, dt64_to_day_of_week


class TestDiscreteTimeApprox:

    @pytest.mark.parametrize('rate, timestep, expected', [
        (1., 1., 1.),
        # output captured from a model-scoped test on BruteForceSEIR
        (0.09118541, 2.0, 0.046682326451980116),
        (0.12820513, 2.0, 0.06630044203966134),
        (0.169492, 2.0, 0.08867788696964007),
        (0.25, 2.0, 0.1339745962155614),
        (0.34482759, 2.0, 0.1905727904084854),
        (0.43478261, 2.0, 0.24819059878496463),
    ])
    def test_py_cy_same(self, rate, timestep, expected):
        py_result = py_dta(rate=rate, timestep=timestep)
        cy_result = cy_dta(rate=rate, timestep=timestep)

        rel_tol = 1e-7
        assert isclose(py_result, cy_result, rel_tol=rel_tol)
        assert isclose(py_result, expected, rel_tol=rel_tol)
        assert isclose(cy_result, expected, rel_tol=rel_tol)


class TestDatetimeUtils:

    @pytest.mark.parametrize('arg, expected', (
        (np.datetime64('2018-01-01'), 0),
        (np.datetime64('2020-11-25'), 2),
    ))
    def test_dt64_to_day_of_week(self, arg, expected):
        result = dt64_to_day_of_week(arg)
        assert result == expected
