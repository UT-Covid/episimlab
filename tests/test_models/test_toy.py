import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import toy
from episimlab.pytest_utils import profiler


@pytest.fixture
def output_vars():
    return {
        'apply_counts_delta__counts': 'step',
        # 'seir__counts_delta_seir': 'step',
        # 'foi__foi': 'step',
        # 'foi__omega': 'step',
    }


@pytest.fixture
def input_vars(config_fp_static):
    return {
        # 'rng__seed_entropy': seed_entropy,
        # 'sto__sto_toggle': sto_toggle
        'read_config__config_fp': config_fp_static
    }


class TestMinimumViable:

    def run_model(self, model, step_clock, input_vars, output_vars):
        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        return input_ds.xsimlab.run(model=model)

    def test_can_report_epis(self, step_clock, input_vars, counts_basic):
        output_vars = {
            # 'apply_counts_delta__counts': 'step',
            'sto__stochastic': 'step',
            'setup_beta__beta': 'step',
            'setup_eta__eta': 'step',
            'setup_gamma__gamma': 'step',
            'setup_mu__mu': 'step',
            'setup_nu__nu': 'step',
            'setup_tau__tau': 'step',
            'setup_rho__rho': 'step',
            'setup_omega__omega': 'step',
            'setup_pi__pi': 'step',
            'setup_sigma__sigma': 'step',
        }
        model = toy.minimum_viable()
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)

        stochastic = result['sto__stochastic']
        logging.debug(f"stochastic: {stochastic}")
        assert stochastic.dtype == bool

        assert 0


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
        return input_ds.xsimlab.run(model=model)

    @profiler()
    @pytest.mark.parametrize('model', (
        toy.slow_seir(),
        toy.slow_seir_cy_foi(),
        toy.cy_seir_cy_foi(),
        toy.cy_seir_w_foi(),
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
        # toy.cy_seir_w_foi(),
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
        toy.cy_seir_w_foi(),
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
        toy.cy_seir_w_foi(),
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
