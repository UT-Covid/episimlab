"""Infrastructure for fitting using linear least squares."""

import os
import attr
import xarray as xr
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from scipy.optimize import least_squares
from episimlab import EPISIMLAB_HOME
from episimlab.models import basic as basic_models
import xsimlab as xs
from xsimlab.model import Model


@attr.s
class FitBetaFromHospHeads:
    """
    """
    data_fp = attr.ib(type=str)
    model = attr.ib(type=Model)
    guess = attr.ib(type=float, default=0.035)

    step_clock = attr.ib(
        type=DatetimeIndex, repr=False,
        default=pd.date_range(start='2/1/2020', end='4/1/2020', freq='12H'))
    config_fp = attr.ib(
        type=str, repr=True,
        default=os.path.join(EPISIMLAB_HOME, 'tests', 'config', 'example_v1.yaml'))
    verbosity = attr.ib(type=int, default=0)
    # extra kwargs to pass to least_squares
    ls_kwargs = attr.ib(type=dict, default=attr.Factory(dict), repr=False)

    @model.default
    def get_default_model(self):
        """Generate default value for attrib `model`."""
        return basic_models.cy_seir_cy_foi().drop_processes(['setup_beta'])
    
    def get_data(self) -> xr.DataArray:
        """Loads hospitalization event data from CSV at `data_fp`.
        Returns the Series as a DataArray.
        """
        df = pd.read_csv(self.data_fp, comment='#')
        df.set_index(['zip_code', 'date'], inplace=True)
        return xr.DataArray.from_series(df['id']).fillna(0.)

    def run(self):
        # get data
        data = self.get_data()
        
        # run fitter
        self.soln = least_squares(
            fun=self.calc_residual,
            x0=self.guess,
            # xtol=1e-8,
            # DEBUG
            xtol=1e-3,
            verbose=self.verbosity,
            # args=(data,),
            # bounds=bounds,
            **self.ls_kwargs
        )
        return self.soln

    def get_in_ds(self) -> xr.Dataset:
        """Generate an input Dataset for the model"""
        return xs.create_setup(
            model=self.model,
            clocks={'step': self.step_clock},
            input_vars={
                'read_config__config_fp': self.config_fp,
                'foi__beta': self.xvars[0]
            },
            output_vars={'apply_counts_delta__counts': 'step'}
        )
    
    def get_out_ds(self) -> xr.Dataset:
        """Run model, returning output Dataset"""
        return (self 
                .get_in_ds() 
                .xsimlab 
                .run(model=self.model, decoding=dict(mask_and_scale=False)))

    def calc_residual(self, xvars) -> float:
        # Store current x variable values in attr
        self.xvars = xvars
        # Run model and pull out counts of Ih compartment over time
        ih_pred = (self
                   .get_out_ds() 
                   .apply_counts_delta__counts 
                   .loc[dict(compartment='Ih')] 
                   .sum(dim=['age_group', 'risk_group', 'vertex']))
        assert len(ih_pred.shape) == 1, (ih_pred.shape, "!= 1")
        assert 'step' in ih_pred.dims, f"'step' is not in {ih_pred.dims}"

        # Calculate residual
        resi = ih_pred - self.data
        return resi