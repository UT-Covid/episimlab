import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi
from ...utils.rng import get_rng


@xs.process
class SetupDefaultGamma(BaseSetupEpi):
    """
    """

    gamma = xs.global_ref('gamma', intent='out')

    def initialize(self):
        self.gamma = self.get_gamma()

    def get_gamma(self) -> xr.DataArray:
        dims = ["compartment"]
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
        return da


@xs.process
class SetupStaticGamma(SetupDefaultGamma):
    """Given a length 3 iterable input `tri_exposed_para`, calculate gamma
    after sampling once from this triangular distibution, at the beginning of
    the simulation.
    """
    tri_h2r = xs.variable(dims=('value'), static=True, intent='in')
    tri_y2r_para = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    def get_gamma(self) -> xr.DataArray:
        dims = ["compartment"]
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [
            self.get_gamma_a(),
            self.get_gamma_y(),
            self.get_gamma_h(),
        ]
        return da

    def get_gamma_h(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_h2r)
        else:
            return 1 / np.mean(self.tri_h2r)

    def get_gamma_y(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_y2r_para)
        else:
            return 1 / np.mean(self.tri_y2r_para)

    def get_gamma_a(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        return self.get_gamma_y()


@xs.process
class SetupDynamicGamma(SetupStaticGamma):
    """Like SetupStaticGamma, but the triangular distibution is sampled
    to calculate gamma at every step.
    """

    def run_step(self):
        self.gamma = self.get_gamma()
