import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from .base import BaseFOI
from .bf_cython_engine import brute_force_FOI


@xs.process
class BruteForceCythonFOI(BaseFOI):
    """A readable, brute force algorithm for calculating force of infection (FOI).
    """
    phi_t = xs.global_ref('phi_t')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')

    def run_step(self):
        """
        """
        # Run in cython, returning a numpy array
        kwargs = dict(
            counts=self.counts.values,
            phi_t=self.phi_t.values,
            omega=self.omega.values,
            beta=self.beta
        )
        foi_arr = brute_force_FOI(**kwargs)

        # Convert the numpy arrray to a DataArray
        self.foi = xr.DataArray(
            data=foi_arr,
            dims=self.FOI_DIMS,
            coords={dim: getattr(self, dim) for dim in self.FOI_DIMS}
        )
