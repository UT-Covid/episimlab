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
    sigma = xs.variable()
    eta = xs.variable()
    mu = xs.variable()
    tau = xs.variable()
    gamma = xs.variable(dims=('compartment'))
    nu = xs.variable(dims=('age_group'))
    pi = xs.variable(dims=('risk_group', 'age_group'))
    rho = xs.variable(dims=('age_group', 'compartment'))

    # Required for force of infection calculation
    # TODO: define these in BaseFOI instead
    beta = xs.variable(intent='in')
    omega = xs.variable(dims=('age_group', 'compartment'), intent='in')

