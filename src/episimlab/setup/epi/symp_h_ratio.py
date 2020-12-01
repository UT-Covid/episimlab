import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultSympHRatio(BaseSetupEpi):
    """Return a default value for symp_h_ratio.
    """

    symp_h_ratio = xs.variable(dims=('age_group'), static=True, intent='out')

    def initialize(self):
        self.symp_h_ratio = [
            0.00070175, 0.00070175, 0.04735258, 0.16329827, 0.25541833
        ]
