"""Infrastructure for fitting using linear least squares."""

import os
import xarray as xr
import pandas as pd
from scipy.optimize import least_squares
from episimlab import EPISIMLAB_HOME
from episimlab.models import basic as basic_models
import xsimlab as xs
from xsimlab.model import Model


def get_ih_actual() -> xr.DataArray:
    # data = 0.
    # da = xr.DataArray(
    #     data=data,
    #     dims=list(),
    #     coords={
    #         'step'
    #     }
    # )
    return 0.


def fit_llsq():
    soln = least_squares(
        fun=run_toy_model,
        x0=0.035,
        # x_scale=x_scale,
        xtol=1e-8,  # default
        verbose=2,
        # bounds=bounds,
        args=(get_ih_actual(),)
    )
    return soln


def run_toy_model(dep_vars, ih_actual) -> float:
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

    # Pull out counts of Ih compartment over time
    ih_pred = (out_ds 
               .apply_counts_delta__counts 
               .loc[dict(compartment='Ih')] 
               .sum(dim=['age_group', 'risk_group', 'vertex']))
    assert len(ih_pred.shape) == 1, (ih_pred.shape, "!= 1")
    assert 'step' in ih_pred.dims, f"'step' is not in {ih_pred.dims}"

    # Calculate residual
    resi = ih_pred - ih_actual
    
    # breakpoint()
    return resi