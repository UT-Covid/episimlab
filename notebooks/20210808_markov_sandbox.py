#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
from itertools import product
import xsimlab as xs
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import networkx as nx


def get_var_dims(process, name) -> tuple:
    """Given process-wrapped class `process`, retrieve the `dims` metadata
    attribute for variable with `name`.
    """
    if not '__xsimlab_cls__' in dir(process):
        raise TypeError(
            f"Expected type 'xsimlab.Process' for arg `process`, received " +
            f"'{type(process)}'"
        )
    var = xs.utils.variables_dict(process).get(name, None)
    if var is None:
        raise AttributeError(f"process '{process}' has no attribute '{name}'")
    return tuple(var.metadata['dims'][0])


def group_dict_by_var(d: dict) -> dict:
    return {k: d[(proc, k)] for (proc, k) in d}


@xs.process
class VaccRate:
    """Provide a `rate_S2V`"""
    rate_S2V = xs.variable(global_name='rate_S2V', groups=['tm'], intent='out')
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
    rate_I2R = xs.variable(global_name='rate_I2R', groups=['tm'], intent='out')
    gamma = xs.variable(global_name='gamma', intent='in')
    state = xs.global_ref('state', intent='in')

    def run_step(self):
        self.rate_I2R = self.gamma * self.I
    
    @property
    def I(self):
        return self.state.loc[dict(compt='I')]


@xs.process
class InitComptGraph:
    """Generate a toy compartment graph"""
    compt_graph = xs.global_ref('compt_graph', intent='out')

    def get_compt_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('V', {"color": "orange"}),
            ('I', {"color": "blue"}),
            ('R', {"color": "green"}),
        ])
        g.add_edges_from([
            ('S', 'I', {"priority": 0, "color": "red"}),
            ('S', 'V', {"priority": 0, "color": "orange"}),
            ('V', 'R', {"priority": None, "color": "orange"}),
            ('I', 'R', {"priority": 1, "color": "blue"}),
        ])
        return g
    
    def vis(self):
        return nx.draw(self.compt_graph)
    
    def initialize(self):
        self.compt_graph = self.get_compt_graph()


@xs.process
class SEIR:
    """Sums subsets of TM (in group `tm`) from many processes to output TM."""
    STATE_DIMS = ('vertex', 'compt', 'age', 'risk')
    _tm_subset = xs.group_dict('tm')
    state = xs.variable(dims=STATE_DIMS, intent='inout', global_name='state')
    tm = xs.variable(dims=STATE_DIMS, intent='out', global_name='tm')
    compt_graph = xs.variable('compt_graph', intent='in', global_name='compt_graph')
    
    def run_step(self):
        """In particular, we need to ensure that `tm_subset` and `tm` refresh
        at every timestep.
        """
        self.tm = xr.zeros_like(self.state)
        self.tm_subset = group_dict_by_var(self._tm_subset)
        self.apply_edges()

    def finalize_step(self):
        self.state += self.tm

    def apply_edges(self) -> None:
        """Iterate over edges in `compt_graph` in ascending order of `priority`.
        Apply each edge to the TM.
        """
        for priority, edges in self.edges_by_priority:
            assert edges, f"no edges with {priority=}"
            k = 1. if np.isinf(priority) else self.calc_k(*edges)
            self.edge_to_tm(*edges, k=k)

    @property
    def edges_by_priority(self) -> tuple[tuple[float, tuple[str, str]]]:
        """Parses the `compt_graph` attribute into tuples of edges sorted
        by edges' `priority` attribute. Basically, only used in the
        `apply_edges` method.
        """
        df = pd.DataFrame(data=self.compt_graph.edges.data('priority'), 
                          columns=['u', 'v', 'priority'])
        df.fillna(-1, inplace=True)
        df.loc[df.priority < 0, 'priority'] = float('inf')
        return tuple(
            tuple((priority, tuple((row.u, row.v) for i, row in grp.iterrows())))
            for priority, grp in df.groupby('priority')
        )

    def calc_k(self, *edges) -> xr.DataArray:
        """Find some scaling factor k such that origin node `u` will not be 
        depleted if all `edges` (u, v) are applied simultaneously. All `edges`
        must share the same origin node u.
        """
        u_set = tuple(set(u for u, v in edges))
        assert len(u_set) == 1, f"edges do not share origin node: {edges=} {u_set=}"
        sum_wt = sum(self.edge_weight(u, v) for u, v in edges)
        u_state = self.state.loc[dict(compt=u_set[0])]
        # set k to infinite if denominator is zero
        k = (u_state / sum_wt).fillna(np.Inf)
        # assert np.all(sum_wt), f"{u_state=}\n{sum_wt=}\n{k=}"
        return xr.ufuncs.minimum(k, xr.ones_like(k))

    def edge_to_tm(self, *edges, k=1.) -> None:
        """Applies to the transition matrix (TM) the weight of directed edges
        (u, v) from compartment `u` to compartment `v`. Scale edge weights by
        `k` to ensure that population of origin node `u` is always non-negative
        after edge weight has been applied.
        """
        for u, v in edges:
            weight = k * self.edge_weight(u, v)
            # print(f"adjusted weight of edge from {u} to {v} is {weight}")
            self.tm.loc[dict(compt=u)] -= weight
            self.tm.loc[dict(compt=v)] += weight

    def edge_weight(self, u, v):
        """Try to find an edge weight for (u, v) from `tm_subset`."""
        key = self.edge_weight_name(u, v)
        if key not in self.tm_subset:
            print(f"WARNING: could not find a weight for transition from {u} to {v} compartment ({key})")
            weight = 0.
        else:
            weight = self.tm_subset[key]
        return weight

    def edge_weight_name(self, u, v) -> str:
        """Key to look for when finding edge weights between nodes `u` and `v`."""
        return f"rate_{u}2{v}"


@xs.process
class InitCoords:
    """Initialize state coordinates"""
    compt = xs.variable(global_name='compt_coords', groups=['coords'], intent='out')
    age = xs.variable(global_name='age_coords', groups=['coords'], intent='out')
    risk = xs.variable(global_name='risk_coords', groups=['coords'], intent='out')
    vertex = xs.variable(global_name='vertex_coords', groups=['coords'], intent='out')
    
    def initialize(self):
        self.compt = ['S', 'I', 'R', 'V'] 
        self.age = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk = ['low', 'high']
        self.vertex = ['Austin', 'Houston', 'San Marcos', 'Dallas']



@xs.process
class InitState:
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
        self.state.loc[dict(compt='I')] = np.array([[20, 20, 20, 20, 20]] * 2).T

    @property
    def dims(self):
        return get_var_dims(SEIR, 'state')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)



@xs.process
class FOI:
    """Calculate force of infection (FOI) with naive
    for looping.
    """
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)
    FOI_DIMS = ('age', 'risk', 'vertex', )
    state = xs.global_ref('state', intent='in')
    beta = xs.variable(global_name='beta', intent='in')
    phi = xs.variable(dims=PHI_DIMS, global_name='phi', intent='in')
    rate_S2I = xs.variable(intent='out', groups=['tm'])
    _coords = xs.group_dict('coords')
    
    def run_step(self):
        self.rate_S2I = self.foi
        # print("foi_bf: ", self.foi_bf)
        # print("foi: ", self.foi)
        # assert 0
        
    @property
    def coords(self):
        return group_dict_by_var(self._coords)
    
    @property
    def foi_dims(self):
        return self.FOI_DIMS
    
    @property
    def foi_coords(self):
        return {dim: self.coords[dim] for dim in self.foi_dims}
    
    @property
    def foi(self) -> xr.DataArray:
        zero_suffix = self.suffixed_dims(self.state[dict(compt=0)], '0')
        one_suffix = self.suffixed_dims(self.state[dict(compt=0)], '1')
        S = self.state.loc[dict(compt='S')].rename(zero_suffix)
        I = self.state.loc[dict(compt='I')].rename(one_suffix)
        N = self.state.sum('compt').rename(one_suffix)
        foi = ((self.beta * self.phi * S * I / N)
               # sum over coords that are not compt
               .sum(one_suffix.values())
               # like .rename({'age0': 'age', 'risk0': 'risk'})
               .rename({v: k for k, v in zero_suffix.items()}))
        return foi
    
    @property
    def foi_bf(self) -> xr.DataArray:
        """Brute force FOI, like BruteForceFOI in Episimlab v1.0"""
        foi = xr.DataArray(data=0., dims=self.foi_dims, coords=self.foi_coords)
        for a0, r0, v0, a1, r1, v1 in product(*[self.age_coords, self.risk_coords, self.vertex_coords, ] * 2):
            i0, i1 = dict(vertex=v0, age=a0, risk=r0), dict(vertex=v1, age=a1, risk=r1)
            phi = self.phi.loc[dict(age0=a0, age1=a1, risk0=r0, risk1=r1, vertex0=v0, vertex1=v1)].values
            S = self.state.loc[dict(compt='S')].loc[i0].values
            I = self.state.loc[dict(compt='I')].loc[i1].values
            N = self.state.loc[i1].sum('compt').values
            foi.loc[i0] += phi * self.beta * S * I / N
        return foi
    
    def suffixed_dims(self, da: xr.DataArray, suffix: str = '0', exclude: list = None) -> xr.DataArray:
        if exclude is None:
            exclude = list()
        return {k: f"{k}{suffix}" for k in da.dims if k not in exclude}



@xs.process
class InitPhi:
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



model = xs.Model({
    'init_phi': InitPhi,
    'init_coords': InitCoords,
    'init_state': InitState,
    'seir': SEIR,
    'foi': FOI,
    'init_compt_graph': InitComptGraph,
    'recovery_rate': RecoveryRate,
    'vacc_rate': VaccRate
})
# model.visualize(show_inputs=True, show_variables=True)


in_ds = xs.create_setup(
    model=model,
    clocks={
        'step': pd.date_range(start='3/1/2020', end='3/15/2020', freq='24H')
    },
    input_vars={
        'foi__beta': 0.08,
        'recovery_rate__gamma': 0.5,
        'vacc_rate__vacc_prop': 0.5,
    },
    output_vars={
        'seir__state': 'step'
    }
)
out_ds = in_ds.xsimlab.run(model=model, decoding=dict(mask_and_scale=False))
plot = out_ds['seir__state'].sum(['age', 'risk', 'vertex']).plot.line(x='step', aspect=2, size=9)
plt.show()
