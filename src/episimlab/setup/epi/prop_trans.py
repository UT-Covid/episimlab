import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultPropTransP(BaseSetupEpi):
    """Return a default value for prop_trans_in_p.
    """

    prop_trans_in_p = xs.variable(dims=(), static=True, intent='out')

    def initialize(self):
        self.hosp_f_ratio = 0.44
