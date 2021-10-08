import xsimlab as xs
import xarray as xr
import numpy as np
import logging


@xs.process
class SetupDefaultTau:
    """Return a default value for tau."""
    tau = xs.global_ref('tau', intent='out')

    def initialize(self):
        self.tau = 0.57


@xs.process
class SetupTauFromAsympRate:
    """Given a static scalar input asymptomatic ratio, calculate tau."""
    asymp_rate = xs.variable(dims=(), static=True, intent='in')
    tau = xs.global_ref('tau', intent='out')

    def initialize(self):
        self.tau = 1 - self.asymp_rate
