import pytest
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultSigma,
    SetupStaticSigmaFromExposedPara,
    SetupDynamicSigmaFromExposedPara
)


class TestSetupSigma:

    @pytest.mark.parametrize('stochastic, expected', (
        (True, 0.3884560589524496),
        (False, 0.3448275862068966),
    ))
    def test_can_setup_static(self, tri_exposed_para, stochastic,
                              seed_state, expected):
        inputs = {
            'tri_exposed_para': tri_exposed_para,
            'stochastic': stochastic,
            'seed_state': seed_state,
        }

        proc = SetupStaticSigmaFromExposedPara(**inputs)
        proc.initialize()
        result = proc.sigma
        assert result == expected

    @pytest.mark.parametrize('stochastic, expected, n_steps', (
        (True, 0.3884560589524496, 1),
        (False, 0.3448275862068966, 1),
        (False, 0.3448275862068966, 10),
        # we expect the same output if the seed_state is the same
        (True, 0.3884560589524496, 10),
    ))
    def test_can_setup_dynamic(self, tri_exposed_para, stochastic, n_steps,
                               seed_state, expected):
        inputs = {
            'tri_exposed_para': tri_exposed_para,
            'stochastic': stochastic,
            'seed_state': seed_state,
        }

        proc = SetupDynamicSigmaFromExposedPara(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.sigma
        assert result == expected
