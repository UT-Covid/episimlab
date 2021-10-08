import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...utils.rng import get_rng


@xs.process
class SetupDefaultRho:
    """Provide a default value for rho_Ia and rho_Iy."""
    rho_Ia = xs.global_ref('rho_Ia', intent='out')
    rho_Iy = xs.global_ref('rho_Iy', intent='out')

    def initialize(self):
        self.rho_Ia = 0.43478261
        self.rho_Iy = 0.43478261


@xs.process
class SetupRhoIa:
    """Calculate rho for compartment Ia."""
    tri_Pa2Ia = xs.variable(global_name='tri_Pa2Ia', static=True, intent='in')
    rho_Ia = xs.global_ref('rho_Ia', intent='out')

    def initialize(self):
        self.rho_Ia = 1 / self.tri_Pa2Ia


@xs.process
class SetupRhoIy:
    """Calculate rho for compartment Iy."""
    tri_Py2Iy = xs.variable(global_name='tri_Py2Iy', static=True, intent='in')
    rho_Iy = xs.global_ref('rho_Iy', intent='out')

    def initialize(self):
        self.rho_Iy = 1 / self.tri_Py2Iy