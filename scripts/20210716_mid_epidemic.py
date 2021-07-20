# !/usr/bin/env python
import argparse
import matplotlib

matplotlib.use('agg')
import pandas as pd
import xarray as xr
import xsimlab as xs
import pandas as pd
from matplotlib import pyplot as plt
from episimlab.models import basic
from episimlab.fit.llsq import FitBetaFromHospHeads
from episimlab.pytest_utils import profiler
from episimlab.partition.partition import Partition2Contact
from episimlab.setup import counts
from episimlab.setup.coords import InitCoordsExpectVertex
from episimlab.setup.counts import InitCountsFromCensusCSV
import dask
from concurrent.futures import ThreadPoolExecutor


# borrowing heavily from tests/test_fit/test_llsq.py
def default_model():
    """Default model for FitBetaFromHospHeads"""
    return (basic
            .cy_seir_cy_foi()
            .drop_processes(['setup_beta'])
            .update_processes({'get_contact_xr': Partition2Contact,
                 'setup_coords': InitCoordsExpectVertex,
                 'setup_counts': InitCountsFromCensusCSV})
    )

def fitter(default_model, config_fp):
    """Returns instantiated fitter"""
    return FitBetaFromHospHeads(
        model=default_model,
        config_fp=config_fp,
        guess=0.035,
        verbosity=2,
        ls_kwargs=dict(xtol=1e-3)
    )

def fit_intra_city(**opts) -> xr.Dataset:

    model = default_model()
    fit = fitter(model, opts['config_fp'])
    soln = fit.run()
    plot = fit.plot()

# todo: run simulation after fitting
def sim_intra_city(**opts) -> xr.Dataset:
    model = (basic
        .partition()
        .drop_processes(['setup_beta'])
        .update_processes(dict(
            get_contact_xr=Partition2Contact,
            setup_counts=InitCountsFromCensusCSV
        ))
    )

    input_vars = opts.copy()
    # Reindex with `process__variable` keys
    input_vars_with_proc = dict()
    for proc, var in model.input_vars:
        assert var in input_vars, f"model requires var {var}, but could not find in input var dict"
        input_vars_with_proc[f"{proc}__{var}"] = input_vars[var]

    # run model
    input_ds = xs.create_setup(
        model=model,
        clocks={'step': pd.date_range(start=opts['start_date'], end=opts['end_date'], freq='24H')},
        input_vars=input_vars_with_proc,
        output_vars=dict(apply_counts_delta__counts='step')
    )
    out_ds = run_model(input_ds, model, n_cores=opts['n_cores'])

    return out_ds


def xr_viz(data_array, sel=dict(), isel=dict(), timeslice=slice(0, None),
           sum_over=['risk_group', 'age_group']):
    """Uses DataArray.plot, which builds on mpl"""
    assert isinstance(data_array, xr.DataArray)
    isel.update({'step': timeslice})
    da = data_array[isel].loc[sel].sum(dim=sum_over)
    _ = da.plot.line(x='step', aspect=2, size=7)


def run_model(input_ds: xr.Dataset, model: xs.Model, n_cores: int) -> xr.Dataset:
    with dask.config.set(pool=ThreadPoolExecutor(n_cores)):
        out_ds = input_ds.xsimlab.run(model=model, parallel=True, decoding=dict(mask_and_scale=False))
    cts = out_ds['apply_counts_delta__counts']
    # breakpoint()

    # plot
    cts.sum(['age_group', 'risk_group', 'vertex']).loc[dict()].plot.line(x='step')
    # plt.show()

    return out_ds

def main(**opts):
    fit_intra_city(**opts)


def get_opts() -> dict:
    """Get options from command line"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="")
    parser.add_argument('--n-cores', default=56,
                        type=str, required=False, help='number of cores to use.')
    parser.add_argument('--config-fp', default='scripts/20210716_mid_epidemic.yaml',
                        type=str, required=False, help='path to YAML configuration file')
    parser.add_argument('--travel-fp', default='data/mid_epidemic/travel0.csv',
                        type=str, required=False, help='path to travel.csv file')
    parser.add_argument('--contacts-fp', type=str, default='data/polymod_contacts.csv',
                        required=False, help='path to contacts.csv file')
    parser.add_argument('--census-counts-csv', type=str,
                        default='data/mid_epidemic/census0.csv',
                        required=False, help='path to file containing populations of ZCTAs')
    parser.add_argument('--start-date', type=str, default='3/11/2020', required=False,
                        help='starting date for the simulation, in string format of pandas.date_range')
    parser.add_argument('--end-date', type=str, default='3/13/2020', required=False,
                        help='end date for the simulation, in string format of pandas.date_range')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opts = get_opts()
    main(**opts)
