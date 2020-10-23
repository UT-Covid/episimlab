import pytest
import xarray as xr
import xsimlab as xs
import logging
from episimlab.models import toy
from episimlab.pytest_utils import profiler


@pytest.fixture
def step_clock():
    return dict(step=range(
        10
    ))

class TestToy:

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
        result = input_ds.xsimlab.run(model=model, parallel=False)
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

