"""Infrastructure for fitting using linear least squares."""

import os
import attr
import xarray as xr
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from scipy.optimize import least_squares
from episimlab import EPISIMLAB_HOME
from episimlab.setup.coords import InitCoordsExpectVertex
from episimlab.models import basic as basic_models
import xsimlab as xs
from xsimlab.model import Model


@attr.s
class FitBetaFromHospHeads:
    """
    """
    model = attr.ib(type=Model)
    data_fp = attr.ib(type=str, default='tests/data/ll_hosp_cumsum.csv')
    guess = attr.ib(type=float, default=0.035)

    config_fp = attr.ib(
        type=str, repr=True,
        default=os.path.join(EPISIMLAB_HOME, 'tests', 'config', 'example_v1.yaml'))
    verbosity = attr.ib(type=int, default=0)
    # extra kwargs to pass to least_squares
    ls_kwargs = attr.ib(type=dict, default=attr.Factory(dict), repr=False)

    @model.default
    def get_default_model(self) -> Model:
        """Generate default value for attrib `model`."""
        model = (basic_models 
                 .cy_seir_cy_foi() 
                 .drop_processes(['setup_beta'])
                 .update_processes({'setup_coords': InitCoordsExpectVertex})
        )
        return model

    def get_data(self) -> xr.DataArray:
        """Loads hospitalization event data from CSV at `data_fp`.
        Returns the Series as a DataArray, setting to attr `data`.
        """
        df = (pd
              .read_csv(self.data_fp, comment='#')
              .rename(columns={'date': 'step', 'zip_code': 'vertex'}))
        df['step'] = pd.to_datetime(df['step'], format="%Y-%m-%d")
        df.set_index(['vertex', 'step'], inplace=True)
        self.data = xr.DataArray.from_series(df['cumsum'])
        return self.data

    def run(self):
        self.get_data()

        # set vertex_labels same as passed data
        self.vertex_labels = self.data.coords['vertex']

        # sum data over vertex axis
        self.data = self.data.sum('vertex')

        # set step_clock to the same daterange in the passed data
        self.step_clock = self.data.coords['step']
        return self.fit()
    
    def fit(self):
        """Run scipy.optimize.least_squares"""
        self.soln = least_squares(
            fun=self.calc_residual,
            x0=self.guess,
            verbose=self.verbosity,
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
                'foi__beta': self.xvars[0],
                'setup_coords__vertex_labels': self.vertex_labels
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
        resi = self.data - ih_pred
        # breakpoint()
        return resi