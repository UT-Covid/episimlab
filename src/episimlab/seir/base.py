import numpy as np
import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords


@xs.process
class BaseSEIR:
    """
    """
    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')

    stochastic = xs.variable(intent='in')
    seed_state = xs.variable(intent='in')

    sigma = xs.variable()
    eta = xs.variable()
    mu = xs.variable()
    tau = xs.variable()
    gamma = xs.variable(dims=('compartment'))
    nu = xs.variable(dims=('age_group'))
    pi = xs.variable(dims=('risk_group', 'age_group'))
    rho = xs.variable(dims=('age_group', 'compartment'))

    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=COUNTS_DIMS,
        static=False,
        intent='out'
    )

    def get_int_per_day(self, step_delta) -> float:
        """
        """
        assert isinstance(step_delta, np.timedelta64)
        return np.timedelta64(1, 'D') / step_delta
