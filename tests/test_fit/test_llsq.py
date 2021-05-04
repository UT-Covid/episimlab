import xarray as xr
from episimlab.fit import llsq


def test_fit_llsq():
    result = llsq.fit_llsq()
    # assert isinstance(result, xr.Dataset)