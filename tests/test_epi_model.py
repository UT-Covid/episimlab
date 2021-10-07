import os
import pytest
import xarray as xr
import xsimlab as xs
import numpy as np
import logging
from episimlab.models import MarkovToy


@pytest.fixture
def model_type():
    return MarkovToy


@pytest.fixture
def config() -> str:
    fp = './tests/data/config/001.yaml'
    assert os.path.isfile(fp), f"Config at {fp} does not exist"
    return fp


class TestEpiModel:

    def test_wo_config(self, model_type):
        """Instantiate model without config"""
        model = model_type()
        _ = model.run()

    def test_w_config(self, model_type, config):
        """Instantiate model with config"""
        model = model_type()
        model.config_fp = config
        r1 = model.run()
        assert model.in_ds['setup_seed__seed_entropy'] == 54321
        assert model.in_ds['setup_sto__sto_toggle'] == 19

        # check override in kwargs
        r2 = model.run(config_fp=config, input_vars=dict(setup_seed__seed_entropy=56789))
        assert model.in_ds['setup_seed__seed_entropy'] == 56789
        assert model.in_ds['setup_sto__sto_toggle'] == 19

        xr.testing.assert_allclose(r1['compt_model__state'], r2['compt_model__state'])

        