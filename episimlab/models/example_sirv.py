#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import xsimlab as xs
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from .epi_model import EpiModel
from ..foi import BaseFOI
from ..compt_model import ComptModel
from ..utils import get_var_dims, group_dict_by_var, visualize_compt_graph
from ..setup.sto import SetupStochasticFromToggle
from ..setup.seed import SeedGenerator


@xs.process
class RecoveryRate:
    """Provide a `rate_I2R`"""
    rate_I2R = xs.variable(global_name='rate_I2R', groups=['edge_weight'], intent='out')
    gamma = xs.variable(global_name='gamma', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_I2R = self.gamma * self.I
    
    @property
    def I(self):
        return self.state.loc[dict(compt='I')]


@xs.process
class SetupComptGraph:
    """A single process in the model. Defines the directed graph `compt_graph`
    that defines the compartments and allowed transitions between them.
    """
    # Reference a variable defined in a different process, and tell the model
    # that this process intends to output this variable.
    compt_graph = xs.global_ref('compt_graph', intent='out')

    def initialize(self):
        """This method is run once at the beginning of the simulation."""
        self.compt_graph = self.get_compt_graph()
    
    def run_step(self):
        """This method is run once at every step of the simulation."""
        pass

    def finalize(self):
        """This method is run once at the end of the simulation."""
        self.visualize()

    def get_compt_graph(self) -> nx.DiGraph:
        """A method that returns a compartment graph as a directed
        graph. Uses the networkx package.
        """
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('I', {"color": "blue"}),
            ('R', {"color": "green"}),
            ('V', {"color": "purple"}), # new
        ])
        g.add_edges_from([
            ('S', 'V', {"priority": 0, "color": "purple"}), # new
            ('S', 'I', {"priority": 0, "color": "red"}),
            ('V', 'I', {"priority": 1, "color": "pink"}), # new
            ('I', 'R', {"priority": 2, "color": "blue"}),
        ])
        return g
    
    def visualize(self):
        """Visualize the compartment graph, saving as a file at a path."""
        return visualize_compt_graph(self.compt_graph)


@xs.process
class SetupCoords:
    """Initialize state coordinates. Imports compartment coordinates from the
    compartment graph.
    """
    compt = xs.index(dims=('compt'), global_name='compt_coords', groups=['coords'])
    age = xs.index(dims=('age'), global_name='age_coords', groups=['coords'])
    risk = xs.index(dims=('risk'), global_name='risk_coords', groups=['coords'])
    vertex = xs.index(dims=('vertex'), global_name='vertex_coords', groups=['coords'])
    compt_graph = xs.global_ref('compt_graph', intent='in')
    
    def initialize(self):
        self.compt = self.compt_graph.nodes
        assert len(self.compt) == 4, len(self.compt)
        self.age = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk = ['low', 'high']
        self.vertex = ['Austin', 'Houston', 'San Marcos', 'Dallas']


@xs.process
class SetupState:
    """Initialize state matrix"""
    _coords = xs.group_dict('coords')
    state = xs.global_ref('state', intent='out')

    def initialize(self):
        self.state = xr.DataArray(
            data=0.,
            dims=self.dims,
            coords=self.coords
        )
        self.state.loc[dict(compt='S')] = 200
        self.state.loc[dict(compt='I')] = 20

    @property
    def dims(self):
        return get_var_dims(ComptModel, 'state')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)


@xs.process
class RateV2I(BaseFOI):
    """A single process in the model. Calculates a force of infection
    for vaccinated persons `rate_V2I`. This process inherits from the
    parent class BaseFOI.
    """
    # Override the default behavior: calculate FOI based on the population
    # of the V compartment, instead of the S compartment
    S_COMPT_LABELS = 'V'
    
    # Like before, we define a variable that we export in this process
    rate_V2I = xs.variable(dims=('age', 'risk', 'vertex'), 
                           global_name='rate_V2I', groups=['edge_weight'], 
                           intent='out')
    
    # We also define an input variable that scales FOI
    vacc_efficacy = xs.variable(global_name='vacc_efficacy', intent='in')
    
    phi = xs.global_ref('phi', intent='in')
    
    def run_step(self):
        """Calculate the `rate_V2I` at every step of the simulation. Here,
        we make use of the `foi` method in the parent process BaseFOI.
        """
        self.rate_V2I = self.foi * (1 - self.vacc_efficacy)


@xs.process
class RateS2V:
    """A single process in the model. Calculates a vaccination rate
    `rate_S2V`. Ingests a `vacc_per_day` with one dimension on `age`.
    """
    vacc_per_day = xs.variable(global_name='vacc_per_day', intent='in',
                               dims=('age')) # new
    rate_S2V = xs.variable(global_name='rate_S2V', groups=['edge_weight'], intent='out')
    
    @xs.runtime(args=['step'])
    def run_step(self, step):
        """Calculate the `rate_S2V` at every step of the simulation.
        Set the rate to zero after step 5.
        """
        if step > 5:
            self.rate_S2V = 0.
        else:
            self.rate_S2V = xr.DataArray(data=self.vacc_per_day, dims=['age']) # new


@xs.process
class FOI(BaseFOI):
    """FOI that provides a `rate_S2I`"""
    TAGS = ('FOI',)
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)
    rate_S2I = xs.variable(intent='out', groups=['edge_weight'])

    def run_step(self):
        self.rate_S2I = self.foi


@xs.process
class SetupPhi:
    """Set value of phi (contacts per unit time)."""
    RANDOM_PHI_DATA = np.array([
        [0.89, 0.48, 0.31, 0.75, 0.07],
        [0.64, 0.69, 0.13, 0.00, 0.05],
        [0.46, 0.58, 0.19, 0.16, 0.11],
        [0.53, 0.36, 0.26, 0.35, 0.13],
        [0.68, 0.70, 0.36, 0.23, 0.28]
    ]) 
    phi = xs.global_ref('phi', intent='out')
    _coords = xs.group_dict('coords')
        
    @property
    def coords(self):
        return group_dict_by_var(self._coords)
    
    @property
    def phi_dims(self):
        return get_var_dims(FOI, 'phi')
    
    @property
    def phi_coords(self):
        return {dim: self.coords[dim.rstrip('01')] for dim in self.phi_dims}
    
    def initialize(self):
        data = self.extend_phi_dims(self.RANDOM_PHI_DATA, self.coords['risk'])
        data = self.extend_phi_dims(data, self.coords['vertex'])
        self.phi = xr.DataArray(data=data, dims=self.phi_dims, coords=self.phi_coords)
    
    def extend_phi_dims(self, data, coords) -> np.ndarray:
        f = lambda data, coords: np.stack([data] * len(coords), axis=-1)
        return f(f(data, coords), coords)


class ExampleSIRV(EpiModel):
    # Optional: include some tags so that future users
    # could sort by model metadata
    TAGS = ('SIRV', 'compartments::4')
    
    # Define all the processes in this model
    PROCESSES = {
        # Core processes
        'compt_model': ComptModel,
        'setup_sto': SetupStochasticFromToggle,
        'setup_seed': SeedGenerator,
        'setup_coords': SetupCoords,
        'setup_state': SetupState,
        'setup_phi': SetupPhi,

        # Edge weight processes from ExampleSIR
        'rate_S2I': FOI,
        'rate_I2R': RecoveryRate,
        
        # Distinct from ExampleSIR
        'setup_compt_graph': SetupComptGraph,
        'rate_S2V': RateS2V,
        'rate_V2I': RateV2I
    }
    
    # Define defaults that can be overwritten by user
    RUNNER_DEFAULTS = {
        'clocks': {
            'step': pd.date_range(start='3/1/2020', end='3/15/2020', freq='24H')
        },
        'input_vars': {
            'sto_toggle': 0, 
            'seed_entropy': 12345,
            'beta': 0.08,
            'gamma': 0.5,
            'vacc_efficacy': 0.9,
            'vacc_per_day': [0, 0, 5, 10, 10]
        },
        'output_vars': {
            'compt_model__state': 'step',
            'rate_V2I__rate_V2I': 'step'
        }
    }
    
    # Define custom plotting methods
    def plot(self):
        """Plot compartment populations over time."""
        return (self
                .out_ds['compt_model__state']
                .sum(['age', 'risk', 'vertex'])
                .plot.line(x='step', aspect=2, size=9))
        
    def plot_vacc(self):
        """Plot population of the vaccinated (V) compartment over time,
        stratified by age group.
        """
        return (self
                .out_ds['compt_model__state']
                .loc[dict(compt='V')]
                .sum(['risk', 'vertex'])
                .plot.line(x='step', aspect=2, size=9))
    
    def plot_rate_V2I(self):
        """Plot incident escape infections (`rate_V2I` over time),
        stratified by age group.
        """
        return (self
                .out_ds['rate_V2I__rate_V2I']
                .sum(['risk', 'vertex'])
                .plot.line(x='step', aspect=2, size=9))