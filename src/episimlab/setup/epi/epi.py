import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class foobar(BaseSetupEpi):
    """
    """
    omega = xs.foreign(BaseFOI, 'omega', intent='out')

    rho = xs.foreign(BaseSEIR, 'rho', intent='out')
    gamma = xs.foreign(BaseSEIR, 'gamma', intent='out')
    sigma = xs.foreign(BaseSEIR, 'sigma', intent='out')
    pi = xs.foreign(BaseSEIR, 'pi', intent='out')
    eta = xs.foreign(BaseSEIR, 'eta', intent='out')
    nu = xs.foreign(BaseSEIR, 'nu', intent='out')
    mu = xs.foreign(BaseSEIR, 'mu', intent='out')
    tau = xs.foreign(BaseSEIR, 'tau', intent='out')

    def initialize(self):
        self.omega = self.get_omega()

    def get_rho(self):
        data = 0.43478261
        dims = ('age_group', 'compartment')
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)

    def get_gamma(self):
        data = 0.
        dims = ['compartment']
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        da = xr.DataArray(data=data, dims=dims, coords=coords)
        da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
        return da

    def get_sigma(self):
        return 0.34482759

    def get_pi(self):
        # pi = np.array(
            # [[5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
             # [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]])
        data = np.array([
            [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
            [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]
        ])
        dims = ('risk_group', 'age_group')
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)

    def get_eta(self):
        return 0.169492

    def get_nu(self):
        dims = ['age_group']
        return xr.DataArray(
            [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128],
            dims=dims,
            coords={k: self.COUNTS_COORDS[k] for k in dims}
        )

    def get_mu(self):
        return 0.12820513

    def get_beta(self):
        return 0.035

    def get_tau(self):
        return 0.57

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
        coords = {k: self.COUNTS_COORDS[k] for k in dims}

        da = xr.DataArray(data=0., dims=dims, coords=coords)
        da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
        assert isinstance(da, xr.DataArray), type(da)
        return da

    def get_rho(self):
        data = 0.43478261
        dims = ('age_group', 'compartment')
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)

    def get_gamma(self):
        data = 0.
        dims = ['compartment']
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        da = xr.DataArray(data=data, dims=dims, coords=coords)
        da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
        return da

    def get_sigma(self):
        return 0.34482759

    def get_pi(self):
        # pi = np.array(
            # [[5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
             # [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]])
        data = np.array([
            [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
            [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]
        ])
        dims = ('risk_group', 'age_group')
        coords = {k: self.COUNTS_COORDS[k] for k in dims}
        return xr.DataArray(data=data, dims=dims, coords=coords)

    def get_eta(self):
        return 0.169492

    def get_nu(self):
        dims = ['age_group']
        return xr.DataArray(
            [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128],
            dims=dims,
            coords={k: self.COUNTS_COORDS[k] for k in dims}
        )

    def get_mu(self):
        return 0.12820513

    def get_beta(self):
        return 0.035

    def get_tau(self):
        return 0.57

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
        coords = {k: self.COUNTS_COORDS[k] for k in dims}

        da = xr.DataArray(data=0., dims=dims, coords=coords)
        da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
        assert isinstance(da, xr.DataArray), type(da)
        return da
