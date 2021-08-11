import xsimlab as xs
import xarray as xr
import numpy as np
from numpy import isinf
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from .base import BaseSEIR
from ..foi.base import BaseFOI
from ..setup.seed import SeedGenerator
from ..setup.sto import InitStochasticFromToggle
from ..utils import discrete_time_approx as py_dta, rng
from ..cy_utils.cy_utils import discrete_time_approx_wrapper as cy_dta


@xs.process
class BruteForceSEIR(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission. Brute force
    algorithm for testing purposes.
    """

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    foi = xs.foreign(BaseFOI, 'foi', intent='in')
    seed_state = xs.foreign(SeedGenerator, 'seed_state', intent='in')
    stochastic = xs.foreign(InitStochasticFromToggle, 'stochastic', intent='in')
    counts_delta_seir = xs.variable(
        groups=['counts_delta'],
        dims=BaseSEIR.COUNTS_DIMS,
        static=False,
        intent='out'
    )

    def get_rng(self):
        return rng.get_rng(seed=self.seed_state)

    def _old_discrete_time_approx(self, rate):
        """
        :param rate: daily rate
        :param timestep: timesteps per day
        :return: rate rescaled by time step
        """
        # if rate >= 1:
            # return np.nan
        # elif timestep == 0:
            # return np.nan
        try:
            rate = float(rate)
            timestep = float(self.int_per_day)
            mod = 1. - (1. - rate)**(1./timestep)
            assert isinstance(rate, Number), (type(rate), rate)
        except AssertionError:
            logging.debug(f"mod: {mod}")
            raise
        else:
            # logging.debug(f"{}")
            return mod

    def discrete_time_approx(self, rate):
        """
        :param rate: daily rate
        :param timestep: timesteps per day
        :return: rate rescaled by time step
        """
        rate = float(rate)
        timestep = float(self.int_per_day)
        val = py_dta(rate=rate, timestep=timestep)
        # logging.debug(f"{rate}, {timestep}, {val}")
        return val

    @xs.runtime(args='step_delta')
    def run_step(self, step_delta):
        """
        """

        # Get a RNG for this timepoint, based off of the uint64 seed
        # at this timepoint
        self.rng = self.get_rng()

        # Get interval per day
        self.int_per_day = self.get_int_per_day(step_delta)

        # Abbreviation for `self.counts_delta`
        self.counts_delta_seir = np.nan * xr.zeros_like(self.counts)

        # Iterate over each vertex
        for v in self.counts.coords['vertex'].values:
            # Iterate over every pair of age-risk categories
            for a, r in product(self.counts.coords['age_group'].values,
                                self.counts.coords['risk_group'].values):

                def idx(compt=None):
                    d = {
                        'vertex': v,
                        'age_group': a,
                        'risk_group': r
                    }
                    # logging.debug(f"d: {d}")
                    if compt is not None:
                        d['compartment'] = compt
                    return d

                def cts(*args, **kwargs):
                    return self.counts.loc[idx(*args, **kwargs)].values


                # Calculate rates of change between each compartment
                rate_S2E = self.foi.loc[{
                    'vertex': v,
                    'age_group': a,
                    'risk_group': r
                }]
                rate_E2P = self.discrete_time_approx(self.sigma) * cts('E')
                rate_Pa2Ia = self.discrete_time_approx(self.rho.loc[{
                    'compartment': 'Ia'
                }]) * cts('Pa')
                rate_Py2Iy = self.discrete_time_approx(self.rho.loc[{
                    'compartment': 'Iy'
                }]) * cts('Py')
                rate_Ia2R = self.discrete_time_approx(self.gamma.loc[{
                    'compartment': 'Ia'
                }]) * cts('Ia')
                rate_Iy2R = self.discrete_time_approx(self.gamma.loc[{
                    'compartment': 'Iy'
                }]) * cts('Iy') * (1 - self.pi.loc[{
                    'age_group': a,
                    'risk_group': r
                }])
                rate_Ih2R = self.discrete_time_approx(self.gamma.loc[{
                    'compartment': 'Ih'
                }]) * cts('Ih') * (1 - self.nu.loc[{
                    'age_group': a,
                }])
                rate_Iy2Ih = self.pi.loc[{
                    'age_group': a,
                    'risk_group': r
                }] * self.discrete_time_approx(self.eta) * cts('Iy')
                rate_Ih2D = self.nu.loc[{
                    'age_group': a
                }] * self.discrete_time_approx(self.mu) * cts('Ih')

                # ------------------- Apply stochasticity ---------------------

                if self.stochastic is True:
                    rate_S2E = self.rng.poisson(rate_S2E)
                    rate_E2P = self.rng.poisson(rate_E2P)
                    rate_Py2Iy = self.rng.poisson(rate_Py2Iy)
                    rate_Pa2Ia = self.rng.poisson(rate_Pa2Ia)
                    rate_Ia2R = self.rng.poisson(rate_Ia2R)
                    rate_Iy2R = self.rng.poisson(rate_Iy2R)
                    rate_Ih2R = self.rng.poisson(rate_Ih2R)
                    rate_Iy2Ih = self.rng.poisson(rate_Iy2Ih)
                    rate_Ih2D = self.rng.poisson(rate_Ih2D)

                if isinf(rate_S2E):
                    rate_S2E = 0
                if isinf(rate_E2P):
                    rate_E2P = 0
                if isinf(rate_Py2Iy):
                    rate_Py2Iy = 0
                if isinf(rate_Pa2Ia):
                    rate_Pa2Ia = 0
                if isinf(rate_Ia2R):
                    rate_Ia2R = 0
                if isinf(rate_Iy2R):
                    rate_Iy2R = 0
                if isinf(rate_Ih2R):
                    rate_Ih2R = 0
                if isinf(rate_Iy2Ih):
                    rate_Iy2Ih = 0
                if isinf(rate_Ih2D):
                    rate_Ih2D = 0

                # ---------------------- Apply deltas -------------------------

                d_S = -rate_S2E
                new_S = cts('S') + d_S
                if new_S < 0:
                    rate_S2E = cts('S')
                    #rate_S2E = 0

                d_E = rate_S2E - rate_E2P
                new_E = cts('E') + d_E
                if new_E < 0:
                    rate_E2P = cts('E') + rate_S2E
                    new_E = 0

                new_E2P = rate_E2P
                new_E2Py = self.tau * rate_E2P
                if new_E2Py < 0:
                    rate_E2P = 0
                    new_E2P = 0
                    new_E2Py = 0

                d_Pa = (1 - self.tau) * rate_E2P - rate_Pa2Ia
                new_Pa = cts('Pa') + d_Pa
                new_Pa2Ia = rate_Pa2Ia
                if new_Pa < 0:
                    rate_Pa2Ia = cts('Pa') + (1 - self.tau) * rate_E2P
                    new_Pa = 0
                    new_Pa2Ia = rate_Pa2Ia

                d_Py = self.tau * rate_E2P - rate_Py2Iy
                new_Py = cts('Py') + d_Py
                new_Py2Iy = rate_Py2Iy
                if new_Py < 0:
                    rate_Py2Iy = cts('Py') + self.tau * rate_E2P
                    new_Py = 0
                    new_Py2Iy = rate_Py2Iy

                new_P2I = new_Pa2Ia + new_Py2Iy

                d_Ia = rate_Pa2Ia - rate_Ia2R
                new_Ia = cts('Ia') + d_Ia
                if new_Ia < 0:
                    rate_Ia2R = cts('Ia') + rate_Pa2Ia
                    new_Ia = 0

                d_Iy = rate_Py2Iy - rate_Iy2R - rate_Iy2Ih
                new_Iy = cts('Iy') + d_Iy
                if new_Iy < 0:
                    rate_Iy2R = (cts('Iy') + rate_Py2Iy) * rate_Iy2R / \
                        (rate_Iy2R + rate_Iy2Ih)
                    rate_Iy2Ih = cts('Iy') + rate_Py2Iy - rate_Iy2R
                    new_Iy = 0

                new_Iy2Ih = rate_Iy2Ih
                if new_Iy2Ih < 0:
                    new_Iy2Ih = 0

                d_Ih = rate_Iy2Ih - rate_Ih2R - rate_Ih2D
                new_Ih = cts('Ih') + d_Ih
                if new_Ih < 0:
                    rate_Ih2R = (cts('Ih') + rate_Iy2Ih) * rate_Ih2R / \
                        (rate_Ih2R + rate_Ih2D)
                    rate_Ih2D = cts('Ih') + rate_Iy2Ih - rate_Ih2R
                    new_Ih = 0

                d_R = rate_Ia2R + rate_Iy2R + rate_Ih2R
                new_R = cts('R') + d_R

                d_D = rate_Ih2D
                new_H2D = rate_Ih2D
                new_D = cts('D') + d_D

                # -------------------- Load into output array -----------------

                self.counts_delta_seir.loc[idx('S')] = new_S - cts('S')
                self.counts_delta_seir.loc[idx('E')] = new_E - cts('E')
                self.counts_delta_seir.loc[idx('Pa')] = new_Pa - cts('Pa')
                self.counts_delta_seir.loc[idx('Py')] = new_Py - cts('Py')
                self.counts_delta_seir.loc[idx('Ia')] = new_Ia - cts('Ia')
                self.counts_delta_seir.loc[idx('Iy')] = new_Iy - cts('Iy')
                self.counts_delta_seir.loc[idx('Ih')] = new_Ih - cts('Ih')
                self.counts_delta_seir.loc[idx('R')] = new_R - cts('R')
                self.counts_delta_seir.loc[idx('D')] = new_D - cts('D')

                self.counts_delta_seir.loc[idx('E2P')] = new_E2P
                self.counts_delta_seir.loc[idx('E2Py')] = new_E2Py
                self.counts_delta_seir.loc[idx('P2I')] = new_P2I
                self.counts_delta_seir.loc[idx('Pa2Ia')] = new_Pa2Ia
                self.counts_delta_seir.loc[idx('Py2Iy')] = new_Py2Iy
                self.counts_delta_seir.loc[idx('Iy2Ih')] = new_Iy2Ih
                self.counts_delta_seir.loc[idx('H2D')] = new_H2D