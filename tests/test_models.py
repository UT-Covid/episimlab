import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import (
    ExampleSIR, ExampleSIRV, PartitionFromTravel, Vaccine)
from episimlab.utils import any_negative


@pytest.mark.parametrize('model_type, sto_toggle', [
    (ExampleSIR, 0), 
    (ExampleSIRV, 0), 
    (PartitionFromTravel, 0),
    (PartitionFromTravel, -1),
    (PartitionFromTravel, 5),
    (Vaccine, 0),
    (Vaccine, -1),
])
def test_model_sanity(model_type, sto_toggle):
    """Tests models with a handful of sanity checks."""
    model = model_type()
    result = model.run(input_vars={'sto_toggle': sto_toggle})
    assert isinstance(result, xr.Dataset)
    state = result['compt_model__state']

    # ensure that no coords are null
    for coord in result.coords.values():
        assert not coord.isnull().any()

    # ensure that the total population has not changed between
    # first and last timepoints
    net_change = (state[dict(step=0)] - state[dict(step=-1)]).sum()
    assert abs(net_change) <= 1e-8

    # ensure that S compt has changed between first and last timesteps
    S_init = state[dict(step=0)].loc[dict(compt="S")]
    S_final = state[dict(step=-1)].loc[dict(compt="S")]
    S_change = (S_final - S_init).sum()
    assert abs(S_change) > 1e-8

    # ensure that there are no negative values in the state at any time
    assert not any_negative(state, raise_err=True)

    # model.plot()
