import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir import base
from ..foi.base import BaseFOI
from ..seir.base import BaseSEIR
from ..setup.coords import SetupDefaultCoords


@xs.process
class SetupStochasticFromToggle:
    """Switches on stochasticity after simulation has run `sto_toggle` steps.
    """
    sto_toggle = xs.variable(static=True, intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='out')

    def initialize(self):
        """Ensures that stochastic is set during initialization"""
        self.run_step(step=0)

    @xs.runtime(args="step")
    def run_step(self, step):
        if self.sto_toggle == -1:
            self.stochastic = False
        elif step >= self.sto_toggle:
            self.stochastic = True
        else:
            self.stochastic = False
        # logging.debug(f"self.stochastic: {self.stochastic}")
        # logging.debug(f"self.sto_toggle: {self.sto_toggle}")
