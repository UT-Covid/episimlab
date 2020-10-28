import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from ..setup.phi import InitPhi
from .base import BaseSEIR
from .bf_cython_w_foi_engine import brute_force_SEIR


@xs.process
class BruteForceCythonWFOI(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission. Brute force
    algorithm for testing purposes.

    TODO: discrete time approximation
    """

    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')
    phi_t = xs.foreign(InitPhi, 'phi_t', intent='in')

    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=BaseSEIR.COUNTS_DIMS,
        static=False,
        intent='out'
    )

    def run_step(self):
        """
        """
        self.counts_delta_seir_arr = brute_force_SEIR(
            counts=self.counts.values,
            phi_t=self.phi_t.values,
            # array type
            rho=self.rho.values,
            gamma=self.gamma.values,
            pi=self.pi.values,
            nu=self.nu.values,
            omega=self.omega.values,
            # float type
            mu=self.mu,
            beta=self.beta,
            sigma=self.sigma,
            tau=self.tau,
            eta=self.eta,
        )

    def finalize_step(self):
        self.counts_delta_seir = xr.DataArray(
            data=self.counts_delta_seir_arr,
            dims=self.counts.dims,
            coords=self.counts.coords
        )
