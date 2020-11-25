import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultMu(BaseSetupEpi):
    """
    """
    mu = xs.foreign(BaseSEIR, 'mu', intent='out')

    def initialize(self):
        self.mu = self.get_mu()

    def get_mu(self):
        return 0.12820513
