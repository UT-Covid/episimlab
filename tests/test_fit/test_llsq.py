import numpy as np
import xarray as xr
from episimlab.fit import llsq
from scipy.optimize.optimize import OptimizeResult


def test_fit_llsq():
    result = llsq.fit_llsq()
    assert isinstance(result, OptimizeResult)
    assert result['success']
    assert result['x'] < 1e-7 