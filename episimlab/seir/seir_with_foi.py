import xsimlab as xs
import xarray as xr

from .base import BaseSEIR
from .bf_cython_engine import seir_with_foi


@xs.process
class SEIRwithFOI(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission."""

    beta = xs.foreign(BaseFOI, 'beta', intent='in')
    omega = xs.foreign(BaseFOI, 'omega', intent='in')
    phi_t = xs.foreign(InitPhi, 'phi_t', intent='in')

    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=BaseSEIR.COUNTS_DIMS,
        static=False,
        intent='out'
    )

    @xs.runtime(args='step_delta')
    def run_step(self, step_delta):
        self.counts_delta_seir_arr = seir_with_foi(
            counts=self.counts.values,
            foi=self.foi.values,
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

