"""Infrastructure for fitting using linear least squares."""

import os
import xarray as xr
import pandas as pd
from scipy.optimize import least_squares
from episimlab import EPISIMLAB_HOME
from episimlab.models import basic as basic_models
import xsimlab as xs
from xsimlab.model import Model


def get_ll_data() -> xr.DataArray:
    data = 0.
    da = xr.DataArray(
        data=data,
        dims=list(),
        coords={
            
        }
    )


def fit_llsq():
    # result = run_toy_model()
    ll_data = get_ll_data()
    soln_full = least_squares(
        fun=run_toy_model,
        x0=0.035,
        # x_scale=x_scale,
        xtol=1e-8,  # default
        # bounds=bounds,
        # args=(ll_data)
    )

    return result


def run_toy_model(dep_vars) -> float:
    beta = dep_vars[0]

    # get config
    config_fp = os.path.join(EPISIMLAB_HOME, 'tests', 'config', 'example_v1.yaml')
    assert os.path.isfile(config_fp)

    # run model
    model = basic_models.cy_seir_cy_foi().drop_processes(['setup_beta'])
    in_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H')
        },
        input_vars={
            'read_config__config_fp': config_fp,
            'foi__beta': beta
        },
        output_vars={
            'apply_counts_delta__counts': 'step'
        }
    )
    # with ResourceProfiler(dt=1.) as rprof:
    out_ds = in_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))
    breakpoint()
    return out_ds