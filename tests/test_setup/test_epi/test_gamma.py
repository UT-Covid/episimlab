import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultGamma,
    SetupStaticGamma,
    SetupDynamicGamma
)


@pytest.fixture()
def tri_h2r():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_y2r_para():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [3.0, 4.0, 5.0]


class TestSetupGamma:

    @pytest.mark.parametrize('stochastic, expected', (
        (True, [0.27216115, 0.27216115, 0.09613157]),
        (False, [0.25, 0.25, 0.091185]),
    ))
    def test_can_setup_static(self, counts_coords, tri_h2r, tri_y2r_para,
                              stochastic, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_h2r': tri_h2r,
            'tri_y2r_para': tri_y2r_para,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupStaticGamma(**inputs)
        proc.initialize()
        result = proc.gamma
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Ih'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('stochastic, expected, n_steps', (
        (True, [0.272161, 0.272161, 0.096132], 1),
        (False, [0.25, 0.25, 0.091185], 1),
        (False, [0.25, 0.25, 0.091185], 10),
        # we expect the same output if the seed_state is the same
        (True, [0.272161, 0.272161, 0.096132], 10),
    ))
    def test_can_setup_dynamic(self, tri_h2r, tri_y2r_para, stochastic,
                               n_steps, counts_coords, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_h2r': tri_h2r,
            'tri_y2r_para': tri_y2r_para,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupDynamicGamma(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.gamma
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Ih'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
