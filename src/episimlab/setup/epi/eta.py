import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultEta(BaseSetupEpi):
    """
    """
    eta = xs.foreign(BaseSEIR, 'eta', intent='out')

    def initialize(self):
        self.eta = self.get_eta()

    def get_eta(self):
        return 0.169492
