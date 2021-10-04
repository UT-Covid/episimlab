import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...utils.rng import get_rng


@xs.process
class SetupDefaultMu:
    """Returns a static, scalar default value for mu.
    """
    mu = xs.global_ref('mu', intent='out')

    def initialize(self):
        self.mu = self.get_mu()

    def get_mu(self):
        return 0.12820513


@xs.process
class SetupStaticMuIh2D:
    """Given a length 3 iterable input `tri_Ih2D`, calculate mu after sampling
    once from this triangular distibution, at the beginning of the simulation.
    """
    tri_Ih2D = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.global_ref('stochastic', intent='in')
    seed_state = xs.global_ref('seed_state', intent='in')
    mu = xs.global_ref('mu', intent='out')

    def initialize(self):
        self.mu = self.get_mu()

    def get_mu(self):
        """Sample from triangular distribution if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_Ih2D)
        else:
            return 1 / np.mean(self.tri_Ih2D)


@xs.process
class SetupDynamicMuIh2D(SetupStaticMuIh2D):
    """Like SetupStaticMuIh2D, but the triangular distibution is sampled
    to calculate mu at every step.
    """

    def run_step(self):
        self.mu = self.get_mu()
