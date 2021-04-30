import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from ..setup.phi import InitPhi, InitPhiGrpMapping
from ..foi.base import BaseFOI
from .base import BaseSEIR
from .bf_cython_engine import brute_force_SEIR


@xs.process
class BruteForceCythonSEIR(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission. Brute force
    algorithm for testing purposes.
    """

    # beta = xs.foreign(BaseFOI, 'beta', intent='in')
    # omega = xs.foreign(BaseFOI, 'omega', intent='in')
    # phi_t = xs.foreign(InitPhi, 'phi_t', intent='in')
    # phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')

    foi = xs.foreign(BaseFOI, 'foi', intent='in')

    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=BaseSEIR.COUNTS_DIMS,
        static=False,
        intent='out'
    )

    @xs.runtime(args='step_delta')
    def run_step(self, step_delta):
        """
        """
        self.counts_delta_seir_arr = brute_force_SEIR(
            # phi_grp_mapping=self.phi_grp_mapping.values,
            counts=self.counts.values,
            foi=self.foi.values,
            # phi_t=self.phi_t.values,
            # array type
            rho=self.rho.values,
            gamma=self.gamma.values,
            pi=self.pi.values,
            nu=self.nu.values,
            # omega=self.omega.values,
            # float type
            mu=self.mu,
            # beta=self.beta,
            sigma=self.sigma,
            tau=self.tau,
            eta=self.eta,
            int_per_day=self.get_int_per_day(step_delta),
            stochastic=self.stochastic,
            int_seed=self.seed_state
        )

    def finalize_step(self):
        self.counts_delta_seir = xr.DataArray(
            data=self.counts_delta_seir_arr,
            dims=self.counts.dims,
            coords=self.counts.coords
        )
