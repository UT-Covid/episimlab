import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultSigma(BaseSetupEpi):
    """
    """

    sigma = xs.foreign(BaseSEIR, 'sigma', intent='out')

    def initialize(self):
        self.sigma = self.get_sigma()

    def get_sigma(self):
        return 0.34482759
