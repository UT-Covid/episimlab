import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultRho,
    SetupStaticRhoFromTri,
    SetupDynamicRhoFromTri
)


@pytest.fixture()
def tri_pa_to_ia():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_py_to_iy():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [3.0, 4.0, 5.0]


class TestSetupRho:

    @pytest.mark.parametrize('stochastic, expected', (
        (True, [0.096132, 0.272161, 0.]),
        (False, [0.091185, 0.25, 0.]),
    ))
    def test_can_setup_static(self, counts_coords, tri_pa_to_ia, tri_py_to_iy,
                              stochastic, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_pa_to_ia': tri_pa_to_ia,
            'tri_py_to_iy': tri_py_to_iy,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupStaticRhoFromTri(**inputs)
        proc.initialize()
        result = proc.rho
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Ih'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('stochastic, expected, n_steps', (
        (True, [0.096132, 0.272161, 0.], 1),
        (False, [0.091185, 0.25, 0.], 1),
        (False, [0.091185, 0.25, 0.], 10),
        # we expect the same output if the seed_state is the same
        (True, [0.096132, 0.272161, 0.], 10),
    ))
    def test_can_setup_dynamic(self, tri_pa_to_ia, tri_py_to_iy, stochastic,
                               n_steps, counts_coords, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_pa_to_ia': tri_pa_to_ia,
            'tri_py_to_iy': tri_py_to_iy,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupDynamicRhoFromTri(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.rho
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Ih'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
