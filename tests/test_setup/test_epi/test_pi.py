import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultPi,
    SetupStaticPi,
    SetupDynamicPi
)

DETERMINISTIC_EXPECTED = 1 / np.array(
    [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
     [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])



class TestSetupPi:

    @pytest.mark.parametrize('expected', (
        (1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])),
        (1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])),
    ))
    def test_can_setup_static(self, counts_coords, symp_h_ratio_w_risk, gamma,
                              eta, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
        })

        proc = SetupStaticPi(**inputs)
        proc.initialize()
        result = proc.pi
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('expected, n_steps', (
        (1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         1),
        (1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         10),
    ))
    def test_can_setup_dynamic(self, symp_h_ratio_w_risk, gamma, eta,
                               n_steps, counts_coords, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
        })

        proc = SetupDynamicPi(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.pi
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
