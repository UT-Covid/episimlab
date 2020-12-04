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
def expected():
    return [0.434783, 0.434783, 0.]


class TestSetupRho:

    def test_can_setup_static(self, counts_coords, tri_pa2ia, tri_py2iy,
                              expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_pa2ia': tri_pa2ia,
            'tri_py2iy': tri_py2iy,
        })

        proc = SetupStaticRhoFromTri(**inputs)
        proc.initialize()
        result = proc.rho
        assert isinstance(result, xr.DataArray)
        result = result.loc[dict(compartment=['Ia', 'Iy', 'Ih'])].values

        logging.debug(f"result: {result}")
        logging.debug(f"expected: {expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('n_steps', (1, 10))
    def test_can_setup_dynamic(self, tri_pa2ia, tri_py2iy,
                               n_steps, counts_coords, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_pa2ia': tri_pa2ia,
            'tri_py2iy': tri_py2iy,
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
