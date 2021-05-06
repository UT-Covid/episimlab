import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize.optimize import OptimizeResult
from episimlab.fit.llsq import FitBetaFromHospHeads
from episimlab.models import basic as basic_models


class TestLeastSqFitters:

    def test_converges_zero(self):
        """Set Ih compartment to zero for entire simulation, and check that beta
        converges to zero
        """
        fitter = FitBetaFromHospHeads(
            model=basic_models.cy_seir_cy_foi().drop_processes(['setup_beta']), 
            guess=0.035, 
            verbosity=2,
            ls_kwargs=dict(xtol=1e-3)
        )
        fitter.step_clock = pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H')
        fitter.data = 0.
        soln = fitter.fit()

        assert isinstance(soln, OptimizeResult)
        assert soln['success']
        assert soln['x'] < 1e-3

    def test_can_fit_cumsum(self):
        """Can fit hospital incidences. Primary use case.
        """
        fitter = FitBetaFromHospHeads(
            model=basic_models.cy_seir_cy_foi().drop_processes(['setup_beta']), 
            data_fp='tests/data/ll_hosp_cumsum.csv',
            guess=0.035, 
            verbosity=2,
            # it has to _kind of_ converge, without wasting time in pytest
            ls_kwargs=dict(xtol=1e-6)
        )
        soln = fitter.run()

        assert isinstance(soln, OptimizeResult)
        assert soln['success']
