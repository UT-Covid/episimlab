#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.random import binomial, hypergeometric
import xarray as xr
import xsimlab as xs
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# from ..setup import epi
from .epi_model import EpiModel
from ..foi import BaseFOI, VaccineFOI
from ..compt_model import ComptModel
from ..utils import (
    get_var_dims, group_dict_by_var, discrete_time_approx as dta,
    IntPerDay, get_rng, any_negative
)
from ..partition.partition import NC2Contact, Contact2Phi
from ..setup.sto import SetupStochasticFromToggle
from ..setup.seed import SeedGenerator
from ..setup.greek import (
    gamma, sigma, rho, mu
)
import logging

logging.basicConfig(level=logging.DEBUG)


@xs.process
class SetupNuDefault:
    """Provide a default value for nu"""
    DIMS = ['age']
    nu = xs.variable(dims=DIMS, global_name='nu', intent='out')
    _coords = xs.group_dict('coords')

    @property
    def dims(self):
        return self.DIMS

    @property
    def coords(self):
        return {k: v for k, v in group_dict_by_var(self._coords).items()
                if k in self.dims}

    def initialize(self):
        self.nu = xr.DataArray(
            [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128],
            dims=self.dims, coords=self.coords)


@xs.process
class SetupPiDefault:
    """Provide a default value for pi"""
    DIMS = ('risk', 'age')
    pi = xs.variable(dims=DIMS, global_name='pi', intent='out')
    _coords = xs.group_dict('coords')

    @property
    def dims(self):
        return self.DIMS

    @property
    def coords(self):
        return {k: v for k, v in group_dict_by_var(self._coords).items()
                if k in self.dims}

    def initialize(self):
        self.pi = xr.DataArray(np.array([
            [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
            [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]]),
            dims=self.dims, coords=self.coords)


@xs.process
class RateS2E(BaseFOI):
    """FOI that provides a `rate_S2E`"""
    TAGS = ('model::ElevenComptV1', 'FOI')
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)
    rate_S2E = xs.variable(intent='out', groups=['tm'])

    @property
    def I(self):
        return self.state.loc[dict(compt=['Ia', 'Iy', 'Pa', 'Py'])]

    @property
    def S(self):
        return self.state.loc[dict(compt='S')]

    def run_step(self):
        self.rate_S2E = self.foi.sum('compt')



@xs.process
class RateS2V:
    """Vaccination dosage model"""
    rate_S2V = xs.variable(global_name='rate_S2V', groups=['tm'], intent='out')
    daily_doses = xs.variable(global_name='daily_doses', intent='in')
    eff_vaccine = xs.variable(global_name='eff_vaccine', intent='in')
    state = xs.global_ref('state', intent='in')

    @property
    def S(self):
        return self.state.loc[dict(compt='S')]

    @property
    def N_minus_S(self):
        return self.state.loc[dict(compt=['V', 'E', 'Ev', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D'])].sum('compt')

    @property
    def N(self):
        return self.state.loc[dict(compt=['S', 'V', 'E', 'Ev', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D'])].sum('compt')

    @property
    def Vs(self):
        return hypergeometric(self.S, self.N_minus_S, self.daily_doses)

    def run_step(self):
        self.rate_S2V = binomial(self.Vs, self.eff_vaccine)


@xs.process
class RateV2Ev(VaccineFOI):
    """FOI that provides a `rate_V2Ev`"""
    TAGS = ('model::ElevenComptV1', 'FOI')
    # reference phi, beta from global environment
    phi = xs.global_ref('phi', intent='in')
    beta = xs.global_ref('beta', intent='in')
    beta_reduction = xs.variable(global_name='beta_reduction', intent='in')
    rate_V2Ev = xs.variable(intent='out', groups=['tm'])

    @property
    def I(self):
        return self.state.loc[dict(compt=['Ia', 'Iy', 'Pa', 'Py'])]

    @property
    def S(self):
        return self.state.loc[dict(compt='V')]

    def run_step(self):
        self.rate_V2Ev = self.foi.sum('compt')


@xs.process
class RateEv2P:
    """Provide a `rate_Ev2P`"""
    rate_Ev2P = xs.variable(global_name='rate_Ev2P', intent='out')
    sigma = xs.variable(global_name='sigma', intent='in')
    state = xs.global_ref('state', intent='in')
    int_per_day = xs.global_ref('int_per_day', intent='in')

    def run_step(self):
        self.rate_Ev2P = dta(self.sigma, self.int_per_day) * self.state.loc[dict(compt='Ev')]


@xs.process
class RateE2P:
    """Provide a `rate_E2P`"""
    rate_E2P = xs.variable(global_name='rate_E2P', intent='out')
    sigma = xs.global_ref('sigma', intent='in')
    state = xs.global_ref('state', intent='in')
    int_per_day = xs.global_ref('int_per_day', intent='in')

    def run_step(self):
        self.rate_E2P = dta(self.sigma, self.int_per_day) * self.state.loc[dict(compt='E')]
        # DEBUG
        assert not any_negative(self.rate_E2P, raise_err=True)


@xs.process
class RateE2Py:
    """Provide a `rate_E2Py`"""
    rate_E2Py = xs.variable(global_name='rate_E2Py', groups=['tm'], intent='out')
    tau = xs.variable(global_name='tau', intent='in')
    rate_E2P = xs.global_ref('rate_E2P', intent='in')

    def run_step(self):
        self.rate_E2Py = self.tau * self.rate_E2P


@xs.process
class RateEv2Py:
    """Provide a `rate_Ev2Py"""
    rate_Ev2Py = xs.variable(global_name='rate_Ev2Py', groups=['tm'], intent='out')
    tau_v = xs.variable(global_name='tau_v', intent='in')
    rate_Ev2P = xs.global_ref('rate_Ev2P', intent='in')

    def run_step(self):
        self.rate_Ev2Py = self.tau_v * self.rate_Ev2P


@xs.process
class RateE2Pa:
    """Provide a `rate_E2Pa`"""
    rate_E2Pa = xs.variable(global_name='rate_E2Pa', groups=['tm'], intent='out')
    tau = xs.variable(global_name='tau', intent='in')
    rate_E2P = xs.global_ref('rate_E2P', intent='in')

    def run_step(self):
        self.rate_E2Pa = (1 - self.tau) * self.rate_E2P
        # DEBUG
        assert not any_negative(self.rate_E2Pa, raise_err=True)


@xs.process
class RateEv2Pa:
    """Provide a `rate_Ev2Pa"""
    rate_Ev2Pa = xs.variable(global_name='rate_Ev2Pa', groups=['tm'], intent='out')
    tau_v = xs.variable(global_name='tau_v', intent='in')
    rate_Ev2P = xs.global_ref('rate_Ev2P', intent='in')

    def run_step(self):
        self.rate_Ev2Pa = (1 - self.tau_v) * self.rate_Ev2P


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
class RateIy2Ih:
    """Provide a `rate_Iy2Ih`"""
    rate_Iy2Ih = xs.variable(global_name='rate_Iy2Ih', groups=['tm'], intent='out')
    eta = xs.variable(global_name='eta', intent='in')
    pi = xs.global_ref('pi', intent='in')
    state = xs.global_ref('state', intent='in')
    int_per_day = xs.global_ref('int_per_day', intent='in')

    def run_step(self):
        self.rate_Iy2Ih = self.pi * dta(self.eta, self.int_per_day) * self.state.loc[dict(compt='Iy')]


@xs.process
class RateIh2D:
    """Provide a `rate_Ih2D`"""
    rate_Ih2D = xs.variable(global_name='rate_Ih2D', groups=['tm'], intent='out')
    mu = xs.variable(global_name='mu', intent='in')
    nu = xs.global_ref('nu', intent='in')
    state = xs.global_ref('state', intent='in')
    int_per_day = xs.global_ref('int_per_day', intent='in')

    def run_step(self):
        self.rate_Ih2D = self.nu * dta(self.mu, self.int_per_day) * self.state.loc[dict(compt='Ih')]


@xs.process
class RateIh2R:
    """Provide a `rate_Ih2R`"""
    rate_Ih2R = xs.variable(global_name='rate_Ih2R', groups=['tm'], intent='out')
    gamma_Ih = xs.variable(global_name='gamma_Ih', intent='in')
    nu = xs.global_ref('nu', intent='in')
    state = xs.global_ref('state', intent='in')
    int_per_day = xs.global_ref('int_per_day', intent='in')

    def run_step(self):
        self.rate_Ih2R = (1 - self.nu) * dta(self.gamma_Ih, self.int_per_day) * self.state.loc[dict(compt='Ih')]


@xs.process
class RateIy2R:
    """Provide a `rate_Iy2R`"""
    rate_Iy2R = xs.variable(global_name='rate_Iy2R', groups=['tm'], intent='out')
    gamma_Iy = xs.variable(global_name='gamma_Iy', intent='in')
    pi = xs.global_ref('pi', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Iy2R = (
                self.gamma_Iy *
                self.state.loc[dict(compt='Iy')] *
                (1 - self.pi))


@xs.process
class RateIa2R:
    """Provide a `rate_Ia2R`"""
    rate_Ia2R = xs.variable(global_name='rate_Ia2R', groups=['tm'], intent='out')
    gamma_Ia = xs.variable(global_name='gamma_Ia', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_Ia2R = self.gamma_Ia * self.state.loc[dict(compt='Ia')]


@xs.process
class SetupComptGraph:
    """Generate an 11-node compartment graph"""
    compt_graph = xs.global_ref('compt_graph', intent='out')

    def get_compt_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('V', {"color": "black"}),
            ('E', {"color": "black"}),
            ('Ev', {"color": "black"}),
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
            ('S', 'V', {"priority": 0}),
            ('V', 'Ev', {"priority": 1}),
            ('Ev', 'Pa', {"priority": 2}),
            ('Ev', 'Py', {"priority": 2}),
            ('E', 'Pa', {"priority": 3}),
            ('E', 'Py', {"priority": 3}),
            ('Pa', 'Ia', {"priority": 4}),
            ('Py', 'Iy', {"priority": 5}),
            ('Ia', 'R', {"priority": 6}),
            ('Iy', 'R', {"priority": 7}),
            ('Iy', 'Ih', {"priority": 7}),
            ('Ih', 'R', {"priority": 8}),
            ('Ih', 'D', {"priority": 8}),
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
        self.compt = ['S', 'V', 'E', 'Ev', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D']
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
        self.state.loc[dict(compt='S')] = np.array([[2000, 2000, 2000, 2000, 2000]] * 2).T
        self.state.loc[dict(compt='Ia')] = np.array([[5, 5, 5, 5, 5]] * 2).T

    @property
    def dims(self):
        return get_var_dims(ComptModel, 'state')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)


@xs.process
class SetupPhi(Contact2Phi):
    """Set value of phi (contacts per unit time)."""
    def initialize_misc_coords(self):
        """Set up coords besides vertex and age group."""
        self.risk = ['low', 'high']
        self.compt = ['S', 'V', 'E', 'Ev', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                      'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                      'Py2Iy', 'Iy2Ih', 'H2D']


@xs.process
class GetContactXR(NC2Contact):
    pass


@xs.process
class PhiLinker:
    """Simple process that passes value of `phi_t` to `phi` for compatibility
    with ESL v1 partition processes.
    """
    phi_t = xs.global_ref('phi_t', intent='in')
    phi = xs.global_ref('phi', intent='out')

    def initialize(self):
        self.phi = self.convert_phi_t()

    def run_step(self):
        self.phi = self.convert_phi_t()

    def convert_phi_t(self):
        return self.phi_t


class Vaccine(EpiModel):
    """Nine-compartment SEIR model with partitioning from Episimlab V1"""
    TAGS = ('SEIR', 'compartments::11', 'contact-partitioning')
    PROCESSES = {
        'setup_compt_graph': SetupComptGraph,
        'compt_model': ComptModel,
        'int_per_day': IntPerDay,
        'get_contact_xr': GetContactXR,
        'setup_phi': SetupPhi,
        #'setup_coords': SetupCoords,
        'setup_state': SetupState,
        'setup_sto': SetupStochasticFromToggle,
        'setup_seed': SeedGenerator,

        # DEBUG: lightweight adapter
        'phi_linker': PhiLinker,

        # calculate greeks used by edge weight processes
        'setup_pi': SetupPiDefault,
        'setup_nu': SetupNuDefault,
        'setup_mu': mu.SetupStaticMuIh2D,
        'setup_gamma_Ih': gamma.SetupGammaIh,
        'setup_gamma_Ia': gamma.SetupGammaIa,
        'setup_gamma_Iy': gamma.SetupGammaIy,
        'setup_sigma': sigma.SetupStaticSigmaFromExposedPara,
        'setup_rho_Ia': rho.SetupRhoIa,
        'setup_rho_Iy': rho.SetupRhoIy,

        # used for RateE2Pa and RateE2Py
        'rate_E2P': RateE2P,
        'rate_Ev2P': RateEv2P,

        # all the expected edge weights
        'rate_S2E': RateS2E,
        'rate_S2V': RateS2V,
        'rate_V2Ev': RateV2Ev,
        'rate_E2Pa': RateE2Pa,
        'rate_Ev2Pa': RateEv2Pa,
        'rate_E2Py': RateE2Py,
        'rate_Ev2Py': RateEv2Py,
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
            'step': pd.date_range(start='3/1/2020', end='5/1/2020', freq='24H')
        },
        input_vars={
            'setup_sto__sto_toggle': 0,
            'setup_seed__seed_entropy': 12345,
            'rate_S2E__beta': 0.35,
            'rate_V2Ev__beta_reduction': 0.1,
            'rate_S2V__daily_doses': 100,
            'rate_S2V__eff_vaccine': 0.8,
            'rate_Iy2Ih__eta': 0.169492,
            'rate_E2Py__tau': 0.57,
            'rate_E2Pa__tau': 0.57,
            'rate_Ev2Py__tau_v': 0.055,
            'rate_Ev2Pa__tau_v': 0.055,
            'setup_rho_Ia__tri_Pa2Ia': 2.3,
            'setup_rho_Iy__tri_Py2Iy': 2.3,
            'setup_sigma__tri_exposed_para': [1.9, 2.9, 3.9],
            'setup_gamma_Ih__tri_Ih2R': [9.4, 10.7, 12.8],
            'setup_gamma_Ia__tri_Iy2R_para': [3.0, 4.0, 5.0],
            'setup_mu__tri_Ih2D': [5.2, 8.1, 10.1],
            'get_contact_xr__contact_da_fp': 'data/20200311_contact_matrix.nc'
        },
        output_vars={
            'compt_model__state': 'step'
        }
    )

    def plot(self, show=True):
        plot = self.out_ds['compt_model__state'].sum(['age', 'risk', 'vertex']).plot.line(x='step', aspect=2, size=9)
        if show:
            plt.show()
