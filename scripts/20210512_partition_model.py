#!/usr/bin/env python
import argparse
import pandas as pd
import xsimlab as xs
from episimlab import apply_counts_delta
from episimlab.models import basic
from episimlab.partition.partition import Partition
from episimlab.setup import coords


def main(**opts):
    # ---------------------------- Define model --------------------------------

    model = (basic
             .cy_seir_cy_foi()
             .update_processes({
                'setup_coords': coords.InitCoordsExceptVertex,
                'setup_phi': Partition
             })
            )

    # ---------------------------- Define inputs -------------------------------

    input_vars = {
        'config_fp': 'tests/config/example_v2.yaml',
        'travel_fp': 'data/20200311_travel.csv',
        'contacts_fp': 'data/polymod_contacts.csv',
    }

    # Reindex with `process__variable` keys
    input_vars_with_proc = dict()
    for proc, var in model.input_vars:
        assert var in input_vars, f"model requires var {var}, but could not find in input var dict"
        input_vars_with_proc[f"{proc}__{var}"] = input_vars[var]

    
    input_ds = xs.create_setup(
        model=model,
        clocks={
            'step': pd.date_range(start='1/1/2018', end='1/3/2018', freq='24H')
        },
        input_vars=input_vars_with_proc,
        output_vars=dict(apply_counts_delta__counts='step')
    )
    # breakpoint()

    # ------------------------------ Run model ---------------------------------

    out_ds = input_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))


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