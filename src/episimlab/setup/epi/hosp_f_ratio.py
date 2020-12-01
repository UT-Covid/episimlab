import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultHospFRatio(BaseSetupEpi):
    """Return a default value for symp_h_ratio.
    """

    hosp_f_ratio = xs.variable(dims=('age_group'), static=True, intent='out')

    def initialize(self):
        self.hosp_f_ratio = [0.04, 0.12365475, 0.03122403, 0.10744644, 0.23157691]
