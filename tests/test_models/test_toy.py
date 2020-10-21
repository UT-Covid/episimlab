import pytest
import xarray as xr
import xsimlab as xs
import logging
from episimlab.models.toy import SingleCitySEIR
from episimlab.pytest_utils import dask_prof

class TestSingleCitySEIR:

    @dask_prof(log_dir='./logs')
    def test_can_run(self, epis, counts_basic, omega, beta):
        assert isinstance(omega, xr.DataArray)
        wrapper = SingleCitySEIR()
        input_vars = dict()
        # input_vars['seir'] = epis
        # input_vars['foi'] = dict(
            # omega=omega,
            # beta=beta
        # )
        # logging.debug(f"input_vars: {input_vars.keys()}")
        output_vars = dict()

        wrapper.input_ds = xs.create_setup(
            model=wrapper.model,
            clocks=dict(step=range(5)),
            input_vars=input_vars,
            output_vars=output_vars
        )
        # input_ds.update(_epis)
        result = wrapper.run()
        assert isinstance(result, xr.Dataset)

