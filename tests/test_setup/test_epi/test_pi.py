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


@pytest.fixture()
def symp_h_ratio_w_risk(counts_coords):
    data = [
        [4.02053589e-04, 3.09130781e-04,
         1.90348188e-02, 4.11412733e-02, 4.87894688e-02],
        [4.02053589e-03, 3.09130781e-03,
         1.90348188e-01, 4.11412733e-01, 4.87894688e-01]
    ]
    dims = ['risk_group', 'age_group']
    return xr.DataArray(
        data=data,
        dims=dims,
        coords={dim: counts_coords[dim] for dim in dims}
    )


class TestSetupPi:

    @pytest.mark.parametrize('stochastic, expected', (
        (True, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])),
        (False, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])),
    ))
    def test_can_setup_static(self, counts_coords, symp_h_ratio_w_risk, gamma,
                              eta, stochastic, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupStaticPi(**inputs)
        proc.initialize()
        result = proc.pi
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('stochastic, expected, n_steps', (
        (True, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         1),
        (False, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         1),
        (False, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         10),
        # we expect the same output if the seed_state is the same
        (True, 1 / np.array(
            [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
             [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]]),
         10),
    ))
    def test_can_setup_dynamic(self, symp_h_ratio_w_risk, gamma, eta, stochastic,
                               n_steps, counts_coords, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
            'stochastic': stochastic,
            'seed_state': seed_state,
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
