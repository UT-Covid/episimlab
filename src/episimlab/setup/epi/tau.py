import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultTau(BaseSetupEpi):
    """
    """

    tau = xs.foreign(BaseSEIR, 'tau', intent='out')

    def initialize(self):
        self.tau = self.get_tau()

    def get_tau(self):
        return 0.57
