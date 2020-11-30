import pytest
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultMu,
    SetupStaticMuFromHtoD,
    SetupDynamicMuFromHtoD
)


@pytest.fixture()
def tri_h2d():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [5.2, 8.1, 10.1]


class TestSetupMu:

    @pytest.mark.parametrize('stochastic, expected', (
        (True, 6.997343840020179),
        (False, 7.8),
    ))
    def test_can_setup_static(self, counts_coords, tri_h2d, stochastic,
                              seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_h2d': tri_h2d,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupStaticMuFromHtoD(**inputs)
        proc.initialize()
        result = proc.mu
        assert result == expected

    @pytest.mark.parametrize('stochastic, expected, n_steps', (
        (True, 6.997343840020179, 1),
        (False, 7.8, 1),
        (False, 7.8, 10),
        # we expect the same output if the seed_state is the same
        (True, 6.997343840020179, 10),
    ))
    def test_can_setup_dynamic(self, tri_h2d, stochastic, n_steps,
                               counts_coords, seed_state, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'tri_h2d': tri_h2d,
            'stochastic': stochastic,
            'seed_state': seed_state,
        })

        proc = SetupDynamicMuFromHtoD(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.mu
        assert result == expected
