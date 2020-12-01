import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi
from ...utils.rng import get_rng


@xs.process
class SetupDefaultRho(BaseSetupEpi):
    """
    """

    rho = xs.foreign(BaseSEIR, 'rho', intent='out')

    def initialize(self):
        self.rho = self.get_rho()

    def get_rho(self):
        dims = ['age_group', 'compartment']
        return xr.DataArray(
            data=0.43478261,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )


@xs.process
class SetupStaticRhoFromTri(SetupDefaultRho):
    """Given a length 3 iterable input `tri_exposed_para`, calculate rho
    after sampling once from this triangular distibution, at the beginning of
    the simulation.
    """
    tri_pa_to_ia = xs.variable(dims=('value'), static=True, intent='in')
    tri_py_to_iy = xs.variable(dims=('value'), static=True, intent='in')
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    def get_rho(self) -> xr.DataArray:
        dims = ["compartment"]
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy'])] = [
            self.get_rho_a(),
            self.get_rho_y(),
        ]
        return da

    def get_rho_y(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_py_to_iy)
        else:
            return 1 / np.mean(self.tri_py_to_iy)

    def get_rho_a(self) -> float:
        """Sample from triangular distributions if stochastic, or return the
        mean if deterministic.
        """
        if self.stochastic is True:
            rng = get_rng(seed=self.seed_state)
            return 1 / rng.triangular(*self.tri_pa_to_ia)
        else:
            return 1 / np.mean(self.tri_pa_to_ia)


@xs.process
class SetupDynamicRhoFromTri(SetupStaticRhoFromTri):
    """Like SetupStaticRho, but the triangular distibution is sampled
    to calculate rho at every step.
    """

    def run_step(self):
        self.rho = self.get_rho()
