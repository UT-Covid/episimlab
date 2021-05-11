import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi
from ...utils.rng import get_rng


@xs.process
class SetupDefaultSigma(BaseSetupEpi):
    """Returns a static, scalar default value for sigma.
    """
    sigma = xs.foreign(BaseSEIR, 'sigma', intent='out')

    def initialize(self):
        self.sigma = self.get_sigma()

    def get_sigma(self):
        return 0.34482759


@xs.process
class SetupStaticSigmaFromExposedPara(SetupDefaultSigma):
    """Given a length 3 iterable input `tri_exposed_para`, calculate sigma
    after sampling once from this triangular distibution, at the beginning of
    the simulation.
    """
    tri_exposed_para = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    def get_sigma(self):
        """Sample from triangular distribution if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_exposed_para)
        else:
            return 1 / np.mean(self.tri_exposed_para)


@xs.process
class SetupDynamicSigmaFromExposedPara(SetupStaticSigmaFromExposedPara):
    """Like SetupStaticSigmaFromExposedPara, but the triangular distibution
    is sampled to calculate sigma at every step.
    """

    def run_step(self):
        self.sigma = self.get_sigma()
