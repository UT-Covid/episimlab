#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import xsimlab as xs
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# from ..setup import epi
from .epi_model import EpiModel
from ..foi import BaseFOI
from ..compt_model import ComptModel
from ..utils import get_var_dims, group_dict_by_var, discrete_time_approx as dta, IntPerDay
import logging
logging.basicConfig(level=logging.DEBUG)


@xs.process
class RateIy2Ih:
    """Provide a `rate_Iy2Ih`"""
    rate_Iy2Ih = xs.variable(global_name='rate_Iy2Ih', groups=['tm'], intent='out')
    pi = xs.variable(global_name='pi', intent='in')
    eta = xs.variable(global_name='eta', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Iy2Ih = self.pi * dta(self.eta) * self.state.loc[dict(compt='Iy')]


@xs.process
class RateIh2D:
    """Provide a `rate_Ih2D`"""
    rate_Ih2D = xs.variable(global_name='rate_Ih2D', groups=['tm'], intent='out')
    mu = xs.variable(global_name='mu', intent='in')
    nu = xs.variable(global_name='nu', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Ih2D = self.nu * dta(self.mu) * self.state.loc[dict(compt='Ih')]


@xs.process
class RateIh2R:
    """Provide a `rate_Ih2R`"""
    rate_Ih2R = xs.variable(global_name='rate_Ih2R', groups=['tm'], intent='out')
    gamma_Ih = xs.variable(global_name='gamma_Ih', intent='in')
    nu = xs.variable(global_name='nu', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Ih2R = (1 - self.nu) * dta(self.gamma_Ih) * self.state.loc[dict(compt='Ih')]


@xs.process
class RatePy2Iy:
    """Provide a `rate_Py2Iy`"""
    rate_Py2Iy = xs.variable(global_name='rate_Py2Iy', groups=['tm'], intent='out')
    rho_Iy = xs.variable(global_name='rho_Iy', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Py2Iy = self.rho_Iy * self.state.loc[dict(compt='Py')]


@xs.process
class RatePa2Ia:
    """Provide a `rate_Pa2Ia`"""
    rate_Pa2Ia = xs.variable(global_name='rate_Pa2Ia', groups=['tm'], intent='out')
    rho_Ia = xs.variable(global_name='rho_Ia', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Pa2Ia = self.rho_Ia * self.state.loc[dict(compt='Pa')]


@xs.process
class RateE2Py:
    """Provide a `rate_E2Py`"""
    rate_E2Py = xs.variable(global_name='rate_E2Py', groups=['tm'], intent='out')
    tau = xs.variable(global_name='tau', intent='in')
    rate_E2P = xs.global_ref('rate_E2P', intent='in')

    def run_step(self):
        self.rate_E2Py = self.tau * self.rate_E2P


@xs.process
class RateE2Pa:
    """Provide a `rate_E2Pa`"""
    rate_E2Pa = xs.variable(global_name='rate_E2Pa', groups=['tm'], intent='out')
    tau = xs.variable(global_name='tau', intent='in')
    rate_E2P = xs.global_ref('rate_E2P', intent='in')

    def run_step(self):
        self.rate_E2Pa = (1 - self.tau) * self.rate_E2P


@xs.process
class RateE2P:
    """Provide a `rate_E2P`"""
    rate_E2P = xs.variable(global_name='rate_E2P', groups=['tm'], intent='out')
    sigma = xs.variable(global_name='sigma', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_E2Py = dta(self.sigma) * self.state.loc[dict(compt='E')]
   

@xs.process
class RateIy2R:
    """Provide a `rate_Iy2R`"""
    rate_Iy2R = xs.variable(global_name='rate_Iy2R', groups=['tm'], intent='out')
    gamma = xs.variable(global_name='gamma', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        raise NotImplemented()
        self.rate_Iy2R = self.gamma * self.Iy


@xs.process
class RateIa2R:
    """Provide a `rate_Ia2R`"""
    rate_Ia2R = xs.variable(global_name='rate_Ia2R', groups=['tm'], intent='out')
    gamma = xs.variable(global_name='gamma', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        raise NotImplemented()
        self.rate_Ia2R = self.gamma * self.Ia


@xs.process
class RateS2E(BaseFOI):
    """FOI that provides a `rate_S2E`"""
    TAGS = ('model::NineComptV1', 'FOI')
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)
    rate_S2E = xs.variable(intent='out', groups=['tm'])

    def run_step(self):
        raise NotImplemented()
        self.rate_S2E = self.foi


@xs.process
class SetupComptGraph:
    """Generate a 9-node compartment graph"""
    compt_graph = xs.global_ref('compt_graph', intent='out')

    def get_compt_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('E', {"color": "black"}),
            ('Pa', {"color": "orange"}),
            ('Py', {"color": "blue"}),
            ('Ia', {"color": "green"}),
            ('Iy', {"color": "purple"}),
            ('Ih', {"color": "yellow"}),
            ('R', {"color": "green"}),
            ('D', {"color": "blue"}),
        ])
        g.add_edges_from([
            ('S', 'E', {"priority": 0}),
            ('E', 'Pa', {"priority": 1}),
            ('E', 'Py', {"priority": 1}),
            ('Pa', 'Ia', {"priority": 2}),
            ('Py', 'Iy', {"priority": 3}),
            ('Ia', 'R', {"priority": 4}),
            ('Iy', 'R', {"priority": 5}),
            ('Iy', 'Ih', {"priority": 5}),
            ('Ih', 'R', {"priority": 6}),
            ('Ih', 'D', {"priority": 6}),
        ])
        return g
    
    def vis(self):
        return nx.draw(self.compt_graph)
    
    def initialize(self):
        self.compt_graph = self.get_compt_graph()


@xs.process
class SetupCoords:
    """Initialize state coordinates"""
    compt = xs.index(dims=('compt'), global_name='compt_coords', groups=['coords'])
    age = xs.index(dims=('age'), global_name='age_coords', groups=['coords'])
    risk = xs.index(dims=('risk'), global_name='risk_coords', groups=['coords'])
    vertex = xs.index(dims=('vertex'), global_name='vertex_coords', groups=['coords'])
    
    def initialize(self):
        self.compt = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D'] 
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
        self.state.loc[dict(compt='S')] = np.array([[200, 200, 200, 200, 200]] * 2).T
        self.state.loc[dict(compt='Ia')] = np.array([[20, 20, 20, 20, 20]] * 2).T

    @property
    def dims(self):
        return get_var_dims(ComptModel, 'state')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)


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



class NineComptV1(EpiModel):
    """Nine-compartment SEIR model with partitioning from Episimlab V1"""
    TAGS = ('SEIR', 'compartments::9')
    # _PROCESSES = dict(
    #     # Random number generator
    #     # rng=seed.SeedGenerator,
    #     # sto=sto.SetupStochasticFromToggle,

    #     # Instantiate coords and counts array
    #     setup_state=SetupState,
    #     setup_coords=SetupCoords,

    #     # Instantiate epidemiological parameters
    #     setup_beta=epi.SetupDefaultBeta,
    #     setup_eta=epi.SetupEtaFromAsympRate,
    #     setup_gamma=epi.SetupStaticGamma,
    #     setup_mu=epi.SetupStaticMuFromHtoD,
    #     setup_nu=epi.SetupStaticNu,
    #     setup_omega=epi.SetupStaticOmega,
    #     setup_pi=epi.SetupStaticPi,
    #     setup_rho=epi.SetupStaticRhoFromTri,
    #     setup_sigma=epi.SetupStaticSigmaFromExposedPara,
    #     setup_tau=epi.SetupTauFromAsympRate,
    #     # setup_phi=Contact2Phi, 
    # )
    PROCESSES = {
        'setup_phi': SetupPhi,
        'setup_coords': SetupCoords,
        'setup_state': SetupState,
        'setup_compt_graph': SetupComptGraph,
        'compt_model': ComptModel,
        'int_per_day': IntPerDay,

        # used for RateE2Pa and RateE2Py
        'rate_E2P': RateE2P,

        # all the expected edge weights
        'rate_Ia2R': RateIa2R,
        'rate_S2E': RateS2E,
        'rate_E2Pa': RateE2Pa,
        'rate_E2Py': RateE2Py,
        'rate_Pa2Ia': RatePa2Ia,
        'rate_Py2Iy': RatePy2Iy,
        'rate_Ia2R': RateIa2R,
        'rate_Iy2R': RateIy2R,
        'rate_Iy2Ih': RateIy2Ih,
        'rate_Ih2R': RateIh2R,
        'rate_Ih2D': RateIh2D,
    }
    RUNNER_DEFAULTS = dict(
        clocks={
            'step': pd.date_range(start='3/1/2020', end='3/15/2020', freq='24H')
        },
        input_vars={
            'rate_S2E__beta': 0.08,
            'rate_Ia2R__gamma': 0.5,
            'rate_E2P__sigma': None, 
            'rate_Ih2D__mu': None, 
            'rate_Ih2D__nu': None, 
            'rate_Py2Iy__rho_Iy': None, 
            'rate_E2Py__tau': None, 
            'rate_Ih2R__gamma_Ih': None, 
            'rate_Ih2R__nu': None, 
            'rate_Iy2R__gamma': None, 
            'rate_E2Pa__tau': None, 
            'rate_Iy2Ih__pi': None, 
            'rate_Iy2Ih__eta': None, 
            'rate_Pa2Ia__rho_Ia': None,
        },
        output_vars={
            'compt_model__state': 'step'
        }
    )

    def plot(self, show=True):
        plot = self.out_ds['compt_model__state'].sum(['age', 'risk', 'vertex']).plot.line(x='step', aspect=2, size=9)
        if show:
            plt.show()
