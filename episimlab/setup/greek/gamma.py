import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...utils.rng import get_rng


@xs.process
class SetupDefaultGamma:
    """Provide default values for `gamma` for compartments Ia, Iy, and Ih."""
    gamma_Ih = xs.global_ref('gamma_Ih', intent='out')
    gamma_Ia = xs.global_ref('gamma_Ia', intent='out')
    gamma_Iy = xs.global_ref('gamma_Iy', intent='out')

    def initialize(self):
        self.gamma_Ih = 0.09118541
        self.gamma_Ia = 0.25


@xs.process
class SetupGammaIh:
    """Draws `gamma` for compartment Ih from a triangular distribution
    defined by 3-length array `tri_h2r`.
    """
    gamma_Ih = xs.global_ref('gamma_Ih', intent='out')
    tri_h2r = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.global_ref('stochastic', intent='in')
    seed_state = xs.global_ref('seed_state', intent='in')

    def get_gamma(self) -> xr.DataArray:
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_h2r)
        else:
            return 1 / np.mean(self.tri_h2r)
    
    def initialize(self):
        self.gamma_Ih = self.get_gamma()


@xs.process
class SetupGammaIa:
    """Draws `gamma` for compartments Ia from a triangular distribution
    defined by 3-length array `tri_y2r_para`.
    """
    gamma_Ia = xs.global_ref('gamma_Ia', intent='out')
    tri_y2r_para = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.global_ref('stochastic', intent='in')
    seed_state = xs.global_ref('seed_state', intent='in')

    def get_gamma_Ia(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_y2r_para)
        else:
            return 1 / np.mean(self.tri_y2r_para)

    def initialize(self):
        """Note that because `seed_state` is same for both rng, gamma_Ia
        and gamma_Iy are exactly the same at each timestep.
        """
        self.gamma_Ia = self.get_gamma_Ia()

@xs.process
class SetupGammaIy:
    """Set gamma for Iy equal to gamma for Ia
    """
    gamma_Ia = xs.global_ref('gamma_Ia', intent='in')
    gamma_Iy = xs.global_ref('gamma_Iy', intent='out')

    def initialize(self):
        self.gamma_Iy = self.gamma_Ia
    
    def run_step(self):
        self.gamma_Iy = self.gamma_Ia
