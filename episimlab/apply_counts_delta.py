import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd


@xs.process
class ApplyCountsDelta:

    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    # also make a discoverable output variable containing dimensions
    # of the counts array. Good for validation
    # counts_dims = xs.variable(
    #     dims=(),
    #     static=True,
    #     intent='out',
    # )
    counts = xs.variable(
        dims=COUNTS_DIMS,
        static=False,
        intent='inout',
    )
    counts_delta = xs.group(name='counts_delta')

    # def initialize(self):
    #     self.counts_dims = self.COUNTS_DIMS

    def aggregate_delta(self, delta_gen):
        try:
            return xr.concat(delta_gen, dim='_delta').sum(dim='_delta')
        except ValueError:
            if len(tuple(delta_gen)) == 0:
                return 0
            else:
                raise

    def finalize_step(self):
        self.counts += self.aggregate_delta(self.counts_delta)
