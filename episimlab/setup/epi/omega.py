import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...foi.base import BaseFOI
from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultOmega(BaseSetupEpi):
    """
    """
    omega = xs.foreign(BaseFOI, 'omega', intent='out')

    def initialize(self):
        self.omega = self.get_omega()

    def get_omega(self):
        # omega_a = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667])
        # omega_y = np.array([1.        , 1.        , 1.        , 1.        , 1.        ])
        # omega_h = np.array([0.        , 0.        , 0.        , 0.        , 0.        ])
        # omega_pa = np.array([0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149])
        # omega_py = np.array([1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724])
        data = np.array([[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
                         [1.        , 1.        , 1.        , 1.        , 1.        ],
                         [0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149],
                         [1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724]])

        dims = ['age_group', 'compartment']
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
        assert isinstance(da, xr.DataArray), type(da)
        return da


@xs.process
class SetupStaticOmega(SetupDefaultOmega):
    """Calculate omega once at the beginning of the simulation.
    """
    prop_trans_in_p = xs.variable(dims=(), static=True, intent='in')
    symp_h_ratio = xs.variable(dims=('age_group'), static=True, intent='in')
    asymp_relative_infect = xs.variable(dims=(), static=True, intent='in')
    gamma = xs.foreign(BaseSEIR, 'gamma', intent='in')
    eta = xs.foreign(BaseSEIR, 'eta', intent='in')
    rho = xs.foreign(BaseSEIR, 'rho', intent='in')
    tau = xs.foreign(BaseSEIR, 'tau', intent='in')

    def get_omega(self) -> xr.DataArray:
        dims = ['age_group', 'compartment']
        da = xr.DataArray(
            data=0.,
            dims=dims,
            coords={k: self.counts_coords[k] for k in dims}
        )
        da.loc[dict(compartment='Ia')] = self.get_omega_a()
        da.loc[dict(compartment='Iy')] = self.get_omega_y()
        _ = self.get_omega_p()
        da.loc[dict(compartment='Pa')] = self.get_omega_pa()
        da.loc[dict(compartment='Py')] = self.get_omega_py()
        return da

    def get_omega_a(self) -> xr.DataArray:
        self.omega_a = xr.DataArray(data=self.asymp_relative_infect,
                                    dims=['age_group'],
                                    coords=dict(age_group=self.age_group))
        return self.omega_a

    def get_omega_y(self) -> xr.DataArray:
        self.omega_y = xr.DataArray(data=1., dims=['age_group'],
                                    coords=dict(age_group=self.age_group))
        return self.omega_y

    def get_omega_pa(self) -> xr.DataArray:
        self.omega_pa = self.omega_p * self.omega_a
        return self.omega_pa

    def get_omega_py(self) -> xr.DataArray:
        self.omega_py = self.omega_p * self.omega_y
        return self.omega_py

    def get_omega_p(self) -> xr.DataArray:
        gamma_y = float(self.gamma.loc[dict(compartment='Iy')])
        gamma_a = float(self.gamma.loc[dict(compartment='Ia')])
        rho_a = float(self.gamma.loc[dict(compartment='Ia')])
        rho_y = float(self.gamma.loc[dict(compartment='Iy')])

        self.omega_p = (
            self.prop_trans_in_p / (1 - self.prop_trans_in_p) *
            (self.tau * self.omega_y * (self.symp_h_ratio / self.eta + (1 - self.symp_h_ratio) / gamma_y) +
            (1 - self.tau) * self.omega_a / gamma_a) /
            ((self.tau * self.omega_y / rho_y + (1 - self.tau) * self.omega_a / rho_a)))

        assert isinstance(self.omega_p, xr.DataArray)
        return self.omega_p


@xs.process
class SetupDynamicOmega(SetupStaticOmega):
    """Like SetupStaticOmega, but recalculate at every step.
    """

    def run_step(self):
        self.omega = self.get_omega()
