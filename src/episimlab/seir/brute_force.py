import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from .base import BaseSEIR
from ..foi.base import BaseFOI


@xs.process
class BruteForceSEIR(BaseSEIR):
    """Calculate change in `counts` due to SEIR transmission. Brute force
    algorithm for testing purposes.

    TODO: discrete time approximation
    """
    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    foi = xs.foreign(BaseFOI, 'foi', intent='in')
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

        # Abbreviation for `self.counts_delta`
        self.counts_delta_seir = xr.zeros_like(self.counts)

        # Iterate over each vertex
        for v in range(self.counts.coords['vertex'].size):
            # Iterate over every pair of age-risk categories
            for a, r in product(self.age_group, self.risk_group):
                # logging.debug([a, r])
                # logging.debug(f"cc: {cc()}")

                # Calculate rates of change between each compartment
                rate_S2E = self.foi.loc[{
                    'vertex': v,
                    'age_group': a,
                    'risk_group': r
                }]
                rate_E2P = self.sigma * cts('E')
                rate_Pa2Ia = self.rho.loc[{
                    'age_group': a,
                    'compartment': 'Ia'
                }] * cts('Pa')
                rate_Py2Iy = self.rho.loc[{
                    'age_group': a,
                    'compartment': 'Iy'
                }] * cts('Py')
                rate_Ia2R = self.gamma.loc[{
                    'compartment': 'Ia'
                }] * cts('Ia')
                rate_Iy2R = self.gamma.loc[{
                    'compartment': 'Iy'
                }] * cts('Iy') * (1 - self.pi.loc[{
                    'age_group': a,
                    'risk_group': r
                }])
                rate_Ih2R = self.gamma.loc[{
                    'compartment': 'Ih'
                }] * cts('Ih') * (1 - self.nu.loc[{
                    'age_group': a,
                }])
                rate_Iy2Ih = self.pi.loc[{
                    'age_group': a,
                    'risk_group': r
                }] * self.eta * cts('Iy')
                rate_Ih2D = self.nu.loc[{
                    'age_group': a
                }] * self.mu * cts('Ih')

                # ---------------------- Apply deltas -------------------------

                d_S = -rate_S2E
                new_S = cts('S') + d_S
                if new_S < 0:
                    rate_S2E = cts('S')
                    rate_S2E = 0

                d_E = rate_S2E - rate_E2P
                new_E = cts('E') + d_E
                if new_E < 0:
                    rate_E2P = cts('E') + rate_S2E
                    new_E = 0

                new_E2P = rate_E2P
                new_E2Py = self.tau * rate_E2P
                if new_E2Py < 0:
                    rate_E2P = 0

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
                    rate_Iy2R = (cts('Iy') + rate_Py2Iy) * rate_Iy2R / (rate_Iy2R + rate_Iy2Ih)
                    rate_Iy2Ih = cts('Iy') + rate_Py2Iy - rate_Iy2R
                    new_Iy = 0

                new_Iy2Ih = rate_Iy2Ih
                if new_Iy2Ih < 0:
                    new_Iy2Ih = 0

                d_Ih = rate_Iy2Ih - rate_Ih2R - rate_Ih2D
                new_Ih = cts('Ih') + d_Ih
                if new_Ih < 0:
                    rate_Ih2R = (cts('Ih') + rate_Iy2Ih) * rate_Ih2R / (rate_Ih2R + rate_Ih2D)
                    rate_Ih2D = cts('Ih') + rate_Iy2Ih - rate_Ih2R
                    new_Ih = 0

                d_R = rate_Ia2R + rate_Iy2R + rate_Ih2R
                new_R = cts('R') + d_R

                d_D = rate_Ih2D
                new_H2D = rate_Ih2D
                new_D = cts('D') + d_D

                # -------------------- Load into output array -----------------

                self.counts_delta_seir.loc[idx('S')] = d_S
                self.counts_delta_seir.loc[idx('E')] = d_E
                self.counts_delta_seir.loc[idx('Pa')] = d_Pa
                self.counts_delta_seir.loc[idx('Py')] = d_Py
                self.counts_delta_seir.loc[idx('Ia')] = d_Ia
                self.counts_delta_seir.loc[idx('Iy')] = d_Iy
                self.counts_delta_seir.loc[idx('Ih')] = d_Ih
                self.counts_delta_seir.loc[idx('R')] = d_R
                self.counts_delta_seir.loc[idx('D')] = d_D

