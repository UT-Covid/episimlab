import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultNu,
    SetupStaticNu,
    SetupDynamicNu
)


@pytest.fixture()
def hosp_f_ratio(counts_coords):
    data = [0.04, 0.12365475, 0.03122403, 0.10744644, 0.23157691]
    dims = ['age_group']
    return xr.DataArray(
        data=data,
        dims=dims,
        coords={dim: counts_coords[dim] for dim in dims}
    )


class TestSetupNu:

    @pytest.mark.parametrize('expected', (
        (1 / np.array([34.74359034, 10.96424693, 44.62289165,
                       12.67944764,  5.66536041])),
    ))
    def test_can_setup_static(self, counts_coords, hosp_f_ratio, gamma, mu,
                              eta, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'hosp_f_ratio': hosp_f_ratio,
            'gamma': gamma,
            'mu': mu,
        })

        proc = SetupStaticNu(**inputs)
        proc.initialize()
        result = proc.nu
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('expected, n_steps', (
        (1 / np.array([34.74359034, 10.96424693, 44.62289165,
                       12.67944764,  5.66536041]),
         1),
        (1 / np.array([34.74359034, 10.96424693, 44.62289165,
                       12.67944764,  5.66536041]),
         10),
    ))
    def test_can_setup_dynamic(self, hosp_f_ratio, gamma, eta, n_steps, mu,
                               counts_coords, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'hosp_f_ratio': hosp_f_ratio,
            'gamma': gamma,
            'mu': mu,
        })

        proc = SetupDynamicNu(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.nu
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
