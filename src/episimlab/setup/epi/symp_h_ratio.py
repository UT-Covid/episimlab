import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from .omega import SetupStaticOmega
from .pi import SetupStaticPi


@xs.process
class SetupDefaultSympHRatio:
    """Return a default value for symp_h_ratio.
    """

    symp_h_ratio = xs.foreign(SetupStaticOmega, 'symp_h_ratio', intent='out')

    def initialize(self):
        self.symp_h_ratio = [0.00070175, 0.00070175, 0.04735258,
                             0.16329827, 0.25541833]


@xs.process
class SetupDefaultSympHRatioWithRisk:
    """Return a default value for symp_h_ratio_w_risk.
    """

    symp_h_ratio_w_risk = xs.foreign(SetupStaticPi, 'symp_h_ratio_w_risk', intent='out')

    def initialize(self):
        self.symp_h_ratio_w_risk = [
            [0.0002791, 0.0002146, 0.0132154, 0.0285634, 0.0338733],
            [0.002791, 0.002146, 0.132154, 0.285634, 0.338733]]
