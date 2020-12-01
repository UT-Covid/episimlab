import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultTau(BaseSetupEpi):
    """Return a default value for tau.
    """
    tau = xs.foreign(BaseSEIR, 'tau', intent='out')

    def initialize(self):
        self.tau = self.get_tau()

    def get_tau(self):
        return 0.57


@xs.process
class SetupTauFromAsympRate(SetupDefaultTau):
    """Given a static scalar input asymptomatic ratio, calculate tau.
    """
    asymp_rate = xs.variable(dims=(), static=True, intent='in')

    def get_tau(self):
        return 1 - self.asymp_rate
