import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from ..setup.phi import InitPhi, InitPhiGrpMapping
from .base import BaseFOI
from .bf_cython_engine import brute_force_FOI

@xs.process
class BruteForceCythonFOI(BaseFOI):
    """A readable, brute force algorithm for calculating force of infection (FOI).
    """
    phi_t = xs.foreign(InitPhi, 'phi_t', intent='in')
    phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')

    def run_step(self):
        """
        """
        self.foi_arr = brute_force_FOI(
            phi_grp_mapping=self.phi_grp_mapping.values,
            counts=self.counts.values,
            phi_t=self.phi_t.values,
            omega=self.omega.values,
            beta=self.beta
        )

    def finalize_step(self):
        self.foi = xr.DataArray(
            data=self.foi_arr,
            dims=self.FOI_DIMS,
            coords={dim: getattr(self, dim) for dim in self.FOI_DIMS}
        )
