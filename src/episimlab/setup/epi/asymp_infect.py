import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultAsympInfect(BaseSetupEpi):
    """Return a default value for asymp_relative_infect.
    """

    asymp_relative_infect = xs.variable(dims=(), static=True, intent='out')

    def initialize(self):
        self.asymp_relative_infect = 0.666666666
