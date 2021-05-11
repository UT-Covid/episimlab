import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize.optimize import OptimizeResult
from xsimlab.model import Model
from episimlab.fit.llsq import FitBetaFromHospHeads
from episimlab.models import basic as basic_models
from episimlab.setup.coords import InitCoordsExpectVertex


@pytest.fixture
def default_model():
    """Default model for FitBetaFromHospHeads"""
    return (basic_models 
            .cy_seir_cy_foi() 
            .drop_processes(['setup_beta'])
            .update_processes({'setup_coords': InitCoordsExpectVertex})
    )


@pytest.fixture
def config_fp():
    return os.path.join('tests', 'config', 'example_v1.yaml')


@pytest.fixture
def fitter(default_model, config_fp):
    """Returns instantiated fitter"""
    return FitBetaFromHospHeads(
        model=default_model,
        config_fp=config_fp,
        guess=0.035, 
        verbosity=2,
        ls_kwargs=dict(xtol=1e-3)
    )

class TestLeastSqFitters:

    @pytest.mark.slow
    def test_can_fit_cumsum(self, fitter):
        """Can fit heads in beds incidences. Primary use case.
        """
        fitter.ls_kwargs = dict(xtol=1e-3)
        soln = fitter.run()

        assert isinstance(soln, OptimizeResult)
        assert soln['success']
        print(f"RMSD: {fitter.rmsd}")
        print(f"Fitted params: {fitter.soln['x']}")

        # can fit
        plot = fitter.plot()
        # breakpoint()
   
    def test_converges_zero(self, fitter):
        """Set Ih compartment to zero for entire simulation, and check that beta
        converges to zero
        """
        fitter.vertex_labels = range(3)
        fitter.ls_kwargs = dict(xtol=1e-3)
        fitter.step_clock = pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H')
        fitter.data = 0.
        soln = fitter.fit()

        assert isinstance(soln, OptimizeResult)
        assert soln['success']
        assert soln['x'] < 1e-3
 
    def test_can_get_default_model(self, fitter):
        model = fitter.get_default_model()
        assert isinstance(model, Model)