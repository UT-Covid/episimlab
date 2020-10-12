import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd

@xs.process
class ApplyCountsDelta:

    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    counts = xs.variable(
        dims=COUNTS_DIMS,
        static=False,
        intent='inout'
    )
    counts_delta = xs.group(name='counts_delta')

    def finalize_step(self):
        self.counts += xr.concat(self.counts_delta, dim='_delta').sum(dim='_delta')

