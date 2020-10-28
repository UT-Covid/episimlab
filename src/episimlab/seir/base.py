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

    sigma = xs.variable()
    eta = xs.variable()
    mu = xs.variable()
    tau = xs.variable()
    gamma = xs.variable(dims=('compartment'))
    nu = xs.variable(dims=('age_group'))
    pi = xs.variable(dims=('risk_group', 'age_group'))
    rho = xs.variable(dims=('age_group', 'compartment'))

