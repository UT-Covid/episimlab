import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultEta(BaseSetupEpi):
    """Return a default value for eta.
    """
    eta = xs.foreign(BaseSEIR, 'eta', intent='out')

    def initialize(self):
        self.eta = self.get_eta()

    def get_eta(self):
        return 0.169492


@xs.process
class SetupEtaFromAsympRate(SetupDefaultEta):
    """Given a static scalar time until hospitalization, calculate eta.
    """
    t_onset_to_h = xs.variable(dims=(), static=True, intent='in')

    def get_eta(self):
        return 1 - self.t_onset_to_h
