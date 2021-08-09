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

@xs.process
class SetupCountsCustom(counts.SetupCountsFromCensusCSV):

    initial_ia = xs.variable(dims=(), intent='in')

    def set_ia(self):
        self.counts.loc[dict(compartment='Ia', risk_group='low')] = self.initial_ia


def partition_from_csv(**opts) -> xr.Dataset:
    model = (basic
             .partition()
             .drop_processes(['setup_beta'])
             .update_processes(dict(
                 get_contact_xr=Partition2Contact, 
                 setup_counts=SetupCountsCustom
             ))
            )
    # breakpoint()

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
    # breakpoint()
    out_ds = run_model(input_ds, model)
    return out_ds


def xr_viz(data_array, sel=dict(), isel=dict(), timeslice=slice(0, None),
           sum_over=['risk_group', 'age_group']):
    """Uses DataArray.plot, which builds on mpl"""
    assert isinstance(data_array, xr.DataArray)
    isel.update({'step': timeslice})
    da = data_array[isel].loc[sel].sum(dim=sum_over)
    _ = da.plot.line(x='step', aspect=2, size=7)


def partition_from_nc() -> xr.Dataset:
    raise NotImplementedError

    model = (basic
             .partition()
             .drop_processes(['setup_beta'])
             .update_processes(dict(setup_counts=counts.SetupCountsFromCensusCSV))
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
    return out_ds


def run_model(input_ds: xr.Dataset, model: xs.Model) -> xr.Dataset:
    out_ds = input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))
    cts = out_ds['apply_counts_delta__counts']
    # breakpoint()

    # plot
    cts.sum(['age_group', 'risk_group', 'vertex']).loc[dict()].plot.line(x='step')
    plt.show()

    return out_ds


def main(**opts):
    partition_from_csv(**opts)


def get_opts() -> dict:
    """Get options from command line"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="")
    # parser.add_argument('', type=argparse.FileType('r'), help="")
    # parser.add_argument("-f", "--func-name", type=str, required=False, help="",
    #                     choices=['partition_from_nc', 'partition_from_csv'],
    #                     default='partition_from_csv')

    parser.add_argument('--config-fp', default='scripts/20210512_partition_model.yaml', 
                        type=str, required=False, help='path to YAML configuration file')
    parser.add_argument('--travel-fp', default='data/20200311_travel.csv', 
                        type=str, required=False, help='path to travel.csv file')
    parser.add_argument('--contacts-fp', type=str, default='data/polymod_contacts.csv', 
                        required=False, help='path to contacts.csv file')
    parser.add_argument('--census-counts-csv', type=str, 
                        default='data/2019_zcta_pop_5_age_groups.csv',
                        required=False, help='path to file containing populations of ZCTAs')
    parser.add_argument('--beta', type=float, default=1., required=False,
                        help='global transmission parameter') 
    parser.add_argument('--initial-ia', type=float, default=50., required=False,
                        help='initial size of the Ia compartment (low risk only)') 
    parser.add_argument('--start-date', type=str, default='3/11/2020', required=False,
                        help='starting date for the simulation, in string format of pandas.date_range') 
    parser.add_argument('--end-date', type=str, default='3/13/2020', required=False,
                        help='end date for the simulation, in string format of pandas.date_range') 
    return vars(parser.parse_args())


if __name__ == '__main__':
    opts = get_opts()
    main(**opts)