import pytest
import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
import logging
import episimlab
from episimlab.seir import (
    base as seir_base,
    brute_force as seir_bf,
    bf_cython as seir_bf_cython,
    bf_cython_w_foi as seir_bf_cython_w_foi
)
from episimlab.foi import (
    base as foi_base,
    brute_force as foi_bf,
    bf_cython as foi_bf_cython,
)
from episimlab.pytest_utils import plotter

VERBOSE = False


@pytest.fixture
def input_vars(seed_entropy, sto_toggle):
    return {
        'rng__seed_entropy': seed_entropy,
        'sto__sto_toggle': sto_toggle
    }


class TestCompareBasicModels:

    # @plotter(flavor='mpl', plotter_kwargs=dict())
    @pytest.mark.parametrize('foi1, seir1', [
        # Python SEIR and FOI
        (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with Python FOI
        (foi_bf.BruteForceFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Python SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with FOI
        (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
    ])
    @pytest.mark.parametrize('foi2, seir2', [
        # Python SEIR and FOI
        # (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with FOI
        # (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
    ])
    def test_seir_foi_combos(self, input_vars, step_clock, foi1, seir1, foi2, seir2):
        # load default model
        model = episimlab.models.toy.slow_seir()

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
        result1 = in_ds.xsimlab.run(model=model1)[out_var_key]
        result2 = in_ds.xsimlab.run(model=model2)[out_var_key]

        # check typing and equality
        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)

        # DEBUG
        try:
            xr.testing.assert_allclose(result1, result2)
        except AssertionError:
            # DEBUG
            diff = result1 - result2
            # 1 if different above threshold
            dw = xr.where(diff <= 1e-5, 1, 0)

            def view(x):
                return x.loc[dict(vertex=1)].sum(
                    ['age_group', 'risk_group']
                )[dict(compartment=slice(None, -7))]
            v1 = view(result1)
            v2 = view(result2)
            compt_labels = diff.coords['compartment']

            if VERBOSE:
                # iterate over coords
                for dim in diff.dims:
                    for c in diff.coords[dim].values:
                        n_diff = dw.loc[{dim: c}].sum().values
                        logging.debug(f"number of different at {dim} == {c}?: {n_diff}")
            raise
