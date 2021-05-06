import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize.optimize import OptimizeResult
from episimlab.fit.llsq import FitBetaFromHospHeads
from episimlab.models import basic as basic_models


def test_fit_llsq():
    model = basic_models.cy_seir_cy_foi().drop_processes(['setup_beta'])
    step_clock = pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H')
    fitter = FitBetaFromHospHeads(
        model=model, 
        data_fp='tests/data/ll_hosp_delta.csv', 
        step_clock=step_clock, 
        guess=0.035, 
        verbosity=2
    )
    fitter.data = 0.
    soln = fitter.run()
    assert isinstance(soln, OptimizeResult)
    assert soln['success']
    assert soln['x'] < 1e-7 