#!/usr/bin/env python
import argparse
import pandas as pd
import xarray as xr
import xsimlab as xs
from matplotlib import pyplot as plt
from episimlab import apply_counts_delta
from episimlab.models import basic
from episimlab.partition.partition import Partition2Contact
from episimlab.setup import coords, counts


def partition_from_csv():
    model = (basic
             .partition()
             .drop_processes(['setup_beta'])
             .update_processes(dict(
                 get_contact_xr=Partition2Contact, 
             ))
            )
    # breakpoint()

    input_vars = {
        'config_fp': 'scripts/20210512_partition_model.yaml',
        'travel_fp': 'data/20200311_travel.csv',
        'contacts_fp': 'data/polymod_contacts.csv',
        'census_counts_csv': 'data/2019_zcta_pop_5_age_groups.csv',
        'beta': 1.
    }
    # Reindex with `process__variable` keys
    input_vars_with_proc = dict()
    for proc, var in model.input_vars:
        assert var in input_vars, f"model requires var {var}, but could not find in input var dict"
        input_vars_with_proc[f"{proc}__{var}"] = input_vars[var]
    
    # run model
    input_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='3/11/2020', end='6/1/2020', freq='24H')
        },
        input_vars=input_vars_with_proc,
        output_vars=dict(apply_counts_delta__counts='step')
    )
    # breakpoint()
    out_ds = run_model(input_ds, model)



def partition_from_nc():
    model = (basic
             .partition()
             .drop_processes(['setup_beta'])
             .update_processes(dict(setup_counts=counts.InitCountsFromCensusCSV))
            )

    input_vars = {
        'config_fp': 'scripts/20210512_partition_model.yaml',
        'contact_da_fp': 'tests/data/20200311_contact_matrix.nc',
        'census_counts_csv': 'data/2019_zcta_pop_5_age_groups.csv',
        'beta': 1.
    }
    # Reindex with `process__variable` keys
    input_vars_with_proc = dict()
    for proc, var in model.input_vars:
        assert var in input_vars, f"model requires var {var}, but could not find in input var dict"
        input_vars_with_proc[f"{proc}__{var}"] = input_vars[var]
    
    # run model
    input_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='3/11/2020', end='6/1/2020', freq='24H')
        },
        input_vars=input_vars_with_proc,
        output_vars=dict(apply_counts_delta__counts='step')
    )
    # breakpoint()
    out_ds = run_model(input_ds, model)


def run_model(input_ds: xr.Dataset, model: xs.Model) -> xr.Dataset:
    out_ds = input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))
    cts = out_ds['apply_counts_delta__counts']
    # breakpoint()

    # plot
    cts.sum(['age_group', 'risk_group', 'vertex']).loc[dict()].plot.line(x='step')
    plt.show()

    return out_ds


def main(func_name, **opts):
    globals()[func_name]()


def get_opts() -> dict:
    """Get options from command line"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="")
    # parser.add_argument('file', type=argparse.FileType('r'), help="")
    parser.add_argument("-f", "--func-name", type=str, required=True, help="",
                        choices=['partition_from_nc', 'partition_from_csv'])
    return vars(parser.parse_args())


if __name__ == '__main__':
    opts = get_opts()
    main(**opts)