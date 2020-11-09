import pytest
import xarray as xr
import xsimlab as xs
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


@pytest.fixture
def step_clock():
    return dict(step=range(
        10
    ))

@pytest.mark.slow
class TestCompareBasicModels:

    @pytest.mark.parametrize('foi1, seir1', [
        # Python SEIR and FOI
        (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Python SEIR with Cython FOI
        (foi_bf.BruteForceFOI, seir_bf_cython.BruteForceCython),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCython),
        # Cython SEIR with FOI
        (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
    ])
    @pytest.mark.parametrize('foi2, seir2', [
        # Python SEIR and FOI
        (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with FOI
        # (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
    ])
    def test_seir_foi_combos(self, step_clock, foi1, seir1, foi2, seir2):
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
        in_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=dict(),
            output_vars=dict(apply_counts_delta__counts='step')
        )

        # run both models
        result1 = in_ds.xsimlab.run(model=model1)['apply_counts_delta__counts']
        result2 = in_ds.xsimlab.run(model=model2)['apply_counts_delta__counts']

        # check typing and equality
        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)
        xr.testing.assert_allclose(result1, result2)


