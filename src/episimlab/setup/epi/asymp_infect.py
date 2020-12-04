import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from .omega import SetupStaticOmega


@xs.process
class SetupDefaultAsympInfect:
    """Return a default value for asymp_relative_infect.
    """

    asymp_relative_infect = xs.foreign(SetupStaticOmega,
                                       'asymp_relative_infect', intent='out')

    def initialize(self):
        self.asymp_relative_infect = 0.666666666
