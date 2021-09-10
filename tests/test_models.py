import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import MarkovToy, NineComptV1


@pytest.mark.parametrize('model_type', [
    MarkovToy, 
    NineComptV1
])
def test_model_sanity(model_type):
    """Tests models with a handful of sanity checks."""
    model = model_type()
    result = model.run_with_defaults()
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

    model.plot()
