import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultOmega,
    SetupStaticOmega,
    SetupDynamicOmega
)


@pytest.fixture()
def expected():
    return np.array(
        [[0.666667, 0.666667, 0.666667, 0.666667, 0.666667],
         [1.      , 1.      , 1.      , 1.      , 1.      ],
         [0.523926, 0.523926, 0.531649, 0.550843, 0.566094],
         [0.785889, 0.785889, 0.797473, 0.826265, 0.849141]])


@pytest.fixture()
def prop_trans_in_p():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.44


@pytest.fixture()
def asymp_relative_infect():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.6666666666666666


class TestSetupOmega:

    def test_can_setup_static(self, counts_coords, prop_trans_in_p, asymp_relative_infect,
                              symp_h_ratio, gamma, eta, rho, tau, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'asymp_relative_infect': asymp_relative_infect,
            'prop_trans_in_p': prop_trans_in_p,
            'symp_h_ratio': symp_h_ratio,
            'gamma': gamma,
            'eta': eta,
            'rho': rho,
            'tau': tau
        })

        proc = SetupStaticOmega(**inputs)
        proc.initialize()
        result = proc.omega
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('n_steps', (1, 10))
    def test_can_setup_dynamic(self, prop_trans_in_p, asymp_relative_infect,
                               symp_h_ratio, gamma, eta, rho, tau,
                               n_steps, counts_coords, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'asymp_relative_infect': asymp_relative_infect,
            'prop_trans_in_p': prop_trans_in_p,
            'symp_h_ratio': symp_h_ratio,
            'gamma': gamma,
            'eta': eta,
            'rho': rho,
            'tau': tau
        })

        proc = SetupDynamicOmega(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.omega
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
