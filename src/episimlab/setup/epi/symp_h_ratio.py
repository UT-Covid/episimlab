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
        self.symp_h_ratio = [0.00070175, 0.00070175, 0.04735258,
                             0.16329827, 0.25541833]


@xs.process
class SetupDefaultSympHRatioWithRisk(BaseSetupEpi):
    """Return a default value for symp_h_ratio_w_risk.
    """

    symp_h_ratio_w_risk = xs.variable(dims=('age_group'), static=True, intent='out')

    def initialize(self):
        self.symp_h_ratio_w_risk = [
            [4.02053589e-04, 3.09130781e-04,
             1.90348188e-02, 4.11412733e-02, 4.87894688e-02],
            [4.02053589e-03, 3.09130781e-03,
             1.90348188e-01, 4.11412733e-01, 4.87894688e-01]
        ]
