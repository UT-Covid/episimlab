"""Infrastructure for fitting using linear least squares."""

import os
import xarray as xr
import pandas as pd
from scipy.optimize import least_squares
from episimlab import EPISIMLAB_HOME
from episimlab.models import basic as basic_models
import xsimlab as xs
from xsimlab.model import Model


def fit_llsq():
    config_fp = os.path.join(EPISIMLAB_HOME, 'tests', 'config', 'example_v1.yaml')
    assert os.path.isfile(config_fp)
    result = run_toy_model(config_fp)
    return result


def run_toy_model(config_fp: str) -> xr.Dataset:
    model = basic_models.cy_seir_cy_foi()

    # Generate input dataset
    in_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H')
        },
        input_vars={
            'read_config__config_fp': config_fp
        },
        output_vars={
            'apply_counts_delta__counts': 'step'
        }
    )

    # Run model
    # with ResourceProfiler(dt=1.) as rprof:
    return in_ds.xsimlab.run(
        model=model,
        decoding=dict(mask_and_scale=False)
    )