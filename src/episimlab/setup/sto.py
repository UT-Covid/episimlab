import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir import base
from ..foi.base import BaseFOI
from ..setup import InitDefaultCoords


@xs.process
class InitStochasticFromToggle:
    """Switches on stochasticity after simulation has run `sto_toggle` steps.
    """
    sto_toggle = xs.variable(static=True, intent='in')
    stochastic = xs.variable(static=False, intent='out')

    @xs.runtime(args="step")
    def run_step(self, step):
        if step >= self.sto_toggle:
            self.stochastic = True
        else:
            self.stochastic = False

