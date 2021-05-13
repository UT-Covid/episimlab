#!/usr/bin/env python
import argparse
import pandas as pd
import xsimlab as xs
from matplotlib import pyplot as plt
from episimlab import apply_counts_delta
from episimlab.models import basic
from episimlab.partition.partition import Partition
from episimlab.setup import coords, counts


def main(**opts):
    # ---------------------------- Define model --------------------------------

    model = (basic
             .partition()
             .drop_processes(['setup_beta'])
             .update_processes(dict(setup_counts=counts.InitCountsFromCensusCSV))
            )

    # ---------------------------- Define inputs -------------------------------

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
    
    input_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='3/11/2020', end='6/1/2020', freq='24H')
        },
        input_vars=input_vars_with_proc,
        output_vars=dict(apply_counts_delta__counts='step')
    )
    # breakpoint()

    # ------------------------------ Run model ---------------------------------

    out_ds = input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))
    cts = out_ds['apply_counts_delta__counts']
    # breakpoint()

    # ------------------------------ Analyze/Plot ------------------------------

    cts.sum(['age_group', 'risk_group', 'vertex']).loc[dict()].plot.line(x='step')
    plt.show()


def get_opts() -> dict:
    """Get options from command line"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="")
    # parser.add_argument('file', type=argparse.FileType('r'), help="")
    parser.add_argument("-f", "--foobar", type=str, required=False, help="")
    return vars(parser.parse_args())


if __name__ == '__main__':
    opts = get_opts()
    main(**opts)