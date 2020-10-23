import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from .base import BaseSEIR
from .bf_cython_w_foi_engine import brute_force_SEIR


@xs.process
class BruteForceCythonWFOI(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission. Brute force
    algorithm for testing purposes.

    TODO: discrete time approximation
    """
    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')

    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=COUNTS_DIMS,
        static=False,
        intent='out'
    )

    def run_step(self):
        """
        """
        self.counts_delta_seir = brute_force_SEIR()
