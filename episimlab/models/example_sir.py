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
class VaccRate:
    """Provide a `rate_S2V`"""
    rate_S2V = xs.variable(global_name='rate_S2V', groups=['edge_weight'], intent='out')
    vacc_prop = xs.variable(global_name="vacc_prop", intent="in")
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        # vaccinate a proportion of S every day
        # self.rate_S2V = self.vacc_prop * self.S

        # vaccinate a flat 20 doses per day
        self.rate_S2V = 20
    
    @property
    def S(self):
        return self.state.loc[dict(compt='S')]


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
    compt_graph = xs.global_ref('compt_graph', intent='out')

    def initialize(self):
        """This method is run once at the beginning of the simulation."""
        self.compt_graph = self.get_compt_graph()

    def finalize(self):
        """This method is run once at the end of the simulation."""
        self.visualize()

    def get_compt_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('I', {"color": "blue"}),
            ('R', {"color": "green"}),
        ])
        g.add_edges_from([
            ('S', 'I', {"priority": 0, "color": "red"}),
            ('I', 'R', {"priority": 1, "color": "blue"}),
        ])
        return g
    
    def visualize(self, path=None):
        """Visualize the compartment graph, saving as a file at `path`"""
        return visualize_compt_graph(self.compt_graph, path=path)


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


@xs.process
class SetupOmega:
    """Set value of omega"""
    omega = xs.global_ref('omega', intent='out')
    _coords = xs.group_dict('coords')
        
    @property
    def coords(self):
        return group_dict_by_var(self._coords)
    
    @property
    def omega_dims(self):
        return get_var_dims(FOI, 'omega')
    
    @property
    def omega_coords(self):
        return {dim: self.coords[dim.rstrip('01')] for dim in self.omega_dims}
    
    def initialize(self):
        da = xr.DataArray(data=1., dims=self.omega_dims, coords=self.omega_coords)
        # da.loc[dict(compt='Ia')] = 0.666666667
        # da.loc[dict(compt='Iy')] = 1.
        # da.loc[dict(compt='Pa')] = [0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149]
        # da.loc[dict(compt='Py')] = [1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724]
        self.omega = da


class ExampleSIR(EpiModel):
    TAGS = ('SIR', 'compartments::3')
    PROCESSES = {
        # core processes
        'compt_model': ComptModel,
        'setup_sto': SetupStochasticFromToggle,
        'setup_seed': SeedGenerator,
        'setup_coords': SetupCoords,
        'setup_state': SetupState,
        'setup_compt_graph': SetupComptGraph,
        'setup_phi': SetupPhi,
        'setup_omega': SetupOmega,

        # calculate weights for edges in the compartment graph
        'rate_S2I': FOI,
        'rate_I2R': RecoveryRate,
    }
    RUNNER_DEFAULTS = dict(
        clocks={
            'step': pd.date_range(start='3/1/2020', end='3/15/2020', freq='24H')
        },
        input_vars={
            'sto_toggle': 0, 
            'seed_entropy': 12345,
            'beta': 0.08,
            'gamma': 0.5,
        },
        output_vars={
            'compt_model__state': 'step'
        }
    )

    def plot(self, show=True):
        plot = self.out_ds['compt_model__state'].sum(['age', 'risk', 'vertex']).plot.line(x='step', aspect=2, size=9)
        if show:
            plt.show()
