import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import toy
from episimlab.pytest_utils import profiler


@pytest.fixture
def step_clock():
    return dict(step=range(
        10
    ))

class TestToyModels:

    @profiler()
    def test_cy_seir(self, epis, counts_basic, step_clock):
        model = toy.cy_seir()
        input_vars = dict()
        output_vars = dict()

        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        result = input_ds.xsimlab.run(model=model)
        assert isinstance(result, xr.Dataset)

    @profiler()
    def test_slow_seir(self, epis, counts_basic, step_clock):
        model = toy.slow_seir()
        input_vars = dict()
        output_vars = dict()

        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        result = input_ds.xsimlab.run(model=model)
        assert isinstance(result, xr.Dataset)


    @profiler(flavor='dask', show_prof=False)
    def test_cy_adj_slow_seir(self, epis, counts_basic, step_clock):
        model = toy.cy_adj_slow_seir()
        input_vars = dict()
        output_vars = dict()

        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        result = input_ds.xsimlab.run(model=model, parallel=True)
        assert isinstance(result, xr.Dataset)


    @profiler()
    def test_cy_adj(self, epis, counts_basic, step_clock):
        model = toy.cy_adj()
        input_vars = dict()
        output_vars = dict()

        input_ds = xs.create_setup(
            model=model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        # input_ds.update(_epis)
        result = input_ds.xsimlab.run(model=model)
        assert isinstance(result, xr.Dataset)

    # TODO
    @pytest.mark.skip
    def test_cy_seir_is_same(self, epis, counts_basic, step_clock):
        """Can the cython SEIR implementation generate same results
        as the "slow" python implementation?
        """
        # Establish inputs and models
        cy_model = toy.cy_seir()
        py_model = toy.slow_seir()
        input_vars = dict()
        output_vars = dict(apply_counts_delta__counts='step')

        # SEIR in cython
        cy_in_ds = xs.create_setup(
            model=cy_model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        cy_result = cy_in_ds.xsimlab.run(model=cy_model)
        assert isinstance(cy_result, xr.Dataset)

        # same inputs, but in python
        py_in_ds = xs.create_setup(
            model=py_model,
            clocks=step_clock,
            input_vars=input_vars,
            output_vars=output_vars
        )
        py_result = py_in_ds.xsimlab.run(model=py_model)
        assert isinstance(cy_result, xr.Dataset)

        # assert equality
        xr.testing.assert_equal(cy_result['apply_counts_delta__counts'],
                                py_result['apply_counts_delta__counts'])

        # non zero?
        assert np.any(cy_result['apply_counts_delta__counts'].values)
        assert np.any(py_result['apply_counts_delta__counts'].values)
