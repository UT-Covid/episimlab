import pytest
import xarray as xr
import xsimlab as xs
from episimlab.models.toy import SingleCitySEIR

class TestSingleCitySEIR:

    def test_can_run(self, epis, counts_basic):
        wrapper = SingleCitySEIR()
        _epis = {f"seir__{k}": epis[k] for k in epis}
        input_vars = dict()
        input_vars.update(_epis)
        output_vars = dict()

        input_ds = xs.create_setup(
            model=wrapper.model,
            clocks=dict(steps=range(70)),
            input_vars=input_vars,
            output_vars=output_vars
        )
        # input_ds.update(_epis)
        # result = model.run()
        assert isinstance(result, xr.Dataset)

