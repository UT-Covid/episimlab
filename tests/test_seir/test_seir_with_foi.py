import pytest
import logging
import xarray as xr
import xsimlab as xs
from episimlab.seir.seir_with_foi import SEIRwithFOI
from episimlab.models.basic import cy_seir_cy_foi, seir_with_foi


class TestSEIRwithFOI:

    def test_can_run_step(self, foi, seed_entropy, stochastic, counts_basic,
                          step_delta, epis, beta, omega, phi_t):
        """
        """
        inputs = {
            'counts': counts_basic,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
            'beta': beta,
            'omega': omega,
            'phi_t': phi_t,
        }
        inputs.update(epis)
        proc = SEIRwithFOI(**inputs)
        proc.run_step(step_delta)

        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

        # TODO
        # assert not result.isnull().any()

    @pytest.mark.parametrize('model1, model2', [
        (cy_seir_cy_foi(), seir_with_foi()),
    ])
    def test_consistent_with_foi_separate(self, step_clock, model1, model2, 
                                          config_fp, config_dict):
        """TestCompareBasicModels.test_seir_foi_combos_deterministic, but
        allows for comparison of models with no `foi` process in the model.
        """
        # generate input variables
        cfg = config_fp(config_dict)
        input_vars = dict(
            read_config__config_fp=cfg,
            setup_coords__config_fp=cfg
        )

        # shared inputs based on the default model
        out_var_key = 'apply_counts_delta__counts'
        in_ds = xs.create_setup(
            model=model1,
            clocks={k: step_clock[k] for k in step_clock},
            input_vars=input_vars,
            output_vars={out_var_key: 'step'}
        )

        # run both models
        result1 = in_ds.xsimlab.run(
            model=model1, decoding=dict(mask_and_scale=False))[out_var_key]
        result2 = in_ds.xsimlab.run(
            model=model2, decoding=dict(mask_and_scale=False))[out_var_key]

        # check typing and equality
        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)
        xr.testing.assert_allclose(result1, result2)