import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import toy
from episimlab.pytest_utils import profiler


@pytest.fixture
def output_vars():
    return {'apply_counts_delta__counts': 'step'}


@pytest.fixture
def input_vars(config_dict, config_fp):
    return dict(read_config__config_fp=config_fp(config_dict))


class TestToyModels:
    """
    TODO: cache model results
    """

    def run_model(self, model, step_clock, input_vars, output_vars):
        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        return input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))

    @profiler()
    @pytest.mark.parametrize('model', (
        toy.slow_seir(),
        toy.slow_seir_cy_foi(),
        toy.cy_seir_cy_foi(),
        toy.cy_adj_slow_seir(),
        # Travel only does not change net S compt
        # toy.cy_adj()
    ))
    def test_can_change_S(self, epis, model, input_vars, counts_basic,
                          output_vars, step_clock):
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)
        counts = result['apply_counts_delta__counts']

        # ensure that S compt has changed between first and last timesteps
        S_init = counts[dict(step=0)].loc[dict(compartment="S")]
        S_final = counts[dict(step=-1)].loc[dict(compartment="S")]
        S_change = (S_final - S_init).sum()
        assert abs(S_change) > 1e-8

    # @profiler()
    @pytest.mark.parametrize('model', (
        toy.slow_seir(),
        toy.slow_seir_cy_foi(),
        toy.cy_seir_cy_foi(),
        # These Cython implementations do not report FOI
        # toy.cy_adj_slow_seir(),
        # toy.cy_adj()
    ))
    def test_non_zero_foi(self, epis, model, input_vars, counts_basic,
                           output_vars, step_clock):
        output_vars['foi__foi'] = 'step'
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)

        # NOTE: will break if no FOI is reported
        # ensure non-zero force of infection at first timepoint
        foi_init = result['foi__foi'][dict(step=1)].sum()
        assert foi_init > 1e-8

    # @profiler()
    @pytest.mark.parametrize('model', (
        toy.slow_seir(),
        toy.slow_seir_cy_foi(),
        toy.cy_seir_cy_foi(),
        toy.cy_adj_slow_seir(),
        toy.cy_adj()
    ))
    def test_constant_pop(self, epis, model, input_vars, counts_basic,
                           output_vars, step_clock):
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)
        counts = result['apply_counts_delta__counts']

        # ensure that the total population has not changed between
        # first and last timepoints
        net_change = (counts[dict(step=0)] - counts[dict(step=-1)]).sum()
        assert abs(net_change) <= 1e-8

    # @profiler()
    @pytest.mark.parametrize('model', (
        toy.slow_seir(),
        toy.slow_seir_cy_foi(),
        toy.cy_seir_cy_foi(),
        toy.cy_adj_slow_seir(),
        toy.cy_adj()
    ))
    def test_non_null_coords(self, epis, model, input_vars, counts_basic,
                             output_vars, step_clock):
        result = self.run_model(model, step_clock, input_vars, output_vars)
        counts = result['apply_counts_delta__counts']

        # ensure that no coords are null
        for coord in result.coords.values():
            assert not coord.isnull().any()
