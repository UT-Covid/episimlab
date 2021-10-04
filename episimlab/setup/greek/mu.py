import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi
from ...utils.rng import get_rng


@xs.process
class SetupDefaultMu(BaseSetupEpi):
    """Returns a static, scalar default value for mu.
    """
    mu = xs.foreign(BaseSEIR, 'mu', intent='out')

    def initialize(self):
        self.mu = self.get_mu()

    def get_mu(self):
        return 0.12820513


@xs.process
class SetupStaticMuFromHtoD(SetupDefaultMu):
    """Given a length 3 iterable input `tri_h2d`, calculate mu after sampling
    once from this triangular distibution, at the beginning of the simulation.
    """
    tri_h2d = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    def get_mu(self):
        """Sample from triangular distribution if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_h2d)
        else:
            return 1 / np.mean(self.tri_h2d)


@xs.process
class SetupDynamicMuFromHtoD(SetupStaticMuFromHtoD):
    """Like SetupStaticMuFromHtoD, but the triangular distibution is sampled
    to calculate mu at every step.
    """

    def run_step(self):
        self.mu = self.get_mu()
