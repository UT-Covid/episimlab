import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...foi.base import BaseFOI
from .base import BaseSetupEpi


@xs.process
class SetupDefaultBeta(BaseSetupEpi):
    """
    """
    beta = xs.foreign(BaseFOI, 'beta', intent='out')

    def initialize(self):
        self.beta = self.get_beta()

    def get_beta(self):
        return 0.035
