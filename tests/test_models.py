import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import basic
from episimlab.pytest_utils import profiler
from episimlab.seir import (
    base as seir_base,
    brute_force as seir_bf,
    bf_cython as seir_bf_cython
)
from episimlab.foi import (
    base as foi_base,
    brute_force as foi_bf,
    bf_cython as foi_bf_cython,
)
from episimlab.pytest_utils import plotter

VERBOSE = False


@pytest.fixture
def output_vars():
    return {'apply_counts_delta__counts': 'step'}


@pytest.fixture
def input_vars(config_dict, config_fp):
    cfg = config_fp(config_dict)
    return dict(
        read_config__config_fp=cfg,
        setup_coords__config_fp=cfg
    )


class TestBasicModels:

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
        # DEBUG: reenable if test_compare_basic fail
        # basic.slow_seir(),
        # basic.slow_seir_cy_foi(),
        # basic.seir_with_foi(),
        basic.cy_seir_cy_foi(),
    ))
    def test_sanity(self, epis, model, input_vars, counts_basic, output_vars,
                    step_clock):
        """Tests models with a handful of sanity checks."""
        result = self.run_model(model, step_clock, input_vars, output_vars)
        assert isinstance(result, xr.Dataset)
        counts = result['apply_counts_delta__counts']

        # ensure that no coords are null
        for coord in result.coords.values():
            assert not coord.isnull().any()

        # ensure that the total population has not changed between
        # first and last timepoints
        net_change = (counts[dict(step=0)] - counts[dict(step=-1)]).sum()
        assert abs(net_change) <= 1e-8

        # ensure that S compt has changed between first and last timesteps
        S_init = counts[dict(step=0)].loc[dict(compartment="S")]
        S_final = counts[dict(step=-1)].loc[dict(compartment="S")]
        S_change = (S_final - S_init).sum()
        assert abs(S_change) > 1e-8

        # NOTE: will break if no FOI is reported
        # ensure non-zero force of infection at first timepoint
        if 'foi__foi' in result:
            foi_init = result['foi__foi'][dict(step=1)].sum()
            assert foi_init > 1e-8

        # check that no phi_grp dims are in the final output dataset (see #4)
        assert not any('phi_grp' in dim for dim in result.dims), \
            (result.dims, "dims contain phi groups")
            

class TestCompareBasicModels:

    # @plotter(flavor='mpl', plotter_kwargs=dict())
    @pytest.mark.slow
    @pytest.mark.parametrize('foi1, seir1', [
        # Python SEIR and FOI
        (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with Python FOI
        (foi_bf.BruteForceFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Python SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf.BruteForceSEIR),
    ])
    @pytest.mark.parametrize('foi2, seir2', [
        # Python SEIR and FOI
        # (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
    ])
    def test_seir_foi_combos_deterministic(self, step_clock, foi1, config_fp,
                                           config_dict, seir1, foi2, seir2):
        """Check that, at model scope, different implementations of a basic
        SEIR model produce consistent results. For instance, a FOI implemented
        in Python should produce the same results as FOI implemented in Cython
        at all timepoints.

        NOTE: some of these tests are not expected to pass. For instance,
        Python and Cython implementations of stochastic SEIR dynamics use
        different random number generators (RNGs), and are expected not to
        produce the same output given the same inputs (including RNG seed).
        There is logic below to catch these expected failures implicitly, so
        that this test should always pass.
        """
        # generate input variables
        cfg = config_fp(config_dict)
        input_vars = dict(
            read_config__config_fp=cfg,
            setup_coords__config_fp=cfg
        )

        # load default model
        model = basic.slow_seir()

        # construct models
        model1 = model.update_processes(dict(
            seir=seir1,
            foi=foi1,
        ))
        model2 = model.update_processes(dict(
            seir=seir2,
            foi=foi2,
        ))

        # shared inputs based on the default model
        out_var_key = 'apply_counts_delta__counts'
        in_ds = xs.create_setup(
            model=model,
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

        # Expect failure if the point of comparison SEIR engine has a Python
        # RNG, and if stochasticity is enabled for this simulation
        failure_expected = bool(
            seir1 is seir_bf.BruteForceSEIR and
            config_dict['sto_toggle'] >= 0
        )
        # implicitly account for results that are not expected to be the same
        # e.g. Python vs. Cython SEIR with different RNGs
        if failure_expected:
            logging.debug("Skipping pytest due to expected discrepancy " +
                          "between Python and Cython RNG")
        else:
            try:
                xr.testing.assert_allclose(result1, result2)
            except AssertionError:
                diff = result2 - result1
                # maximum difference between results
                max_diff = float(abs(diff).max())
                logging.debug(f"max_diff: {max_diff}")
                # where is the max diff
                where_max_diff = diff.where(abs(diff) >= max_diff / 2., drop=True)
                logging.debug(f"where_max_diff: {where_max_diff}")

                raise
