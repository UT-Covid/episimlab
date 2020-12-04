import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from .omega import SetupStaticOmega


@xs.process
class SetupDefaultPropTransP:
    """Return a default value for prop_trans_in_p.
    """

    prop_trans_in_p = xs.foreign(SetupStaticOmega, 'prop_trans_in_p', intent='out')

    def initialize(self):
        self.prop_trans_in_p = 0.44
