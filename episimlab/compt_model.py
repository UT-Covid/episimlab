import logging
import numpy as np
import xarray as xr
import xsimlab as xs
import pandas as pd
from .utils.variable import group_dict_by_var, any_negative, clip_to_zero, edge_weight_name
from .utils.rng import get_rng
from numbers import Number


@xs.process
class ComptModel:
    """Applies the compartmental disease model defined in `compt_graph` to the
    current `state` of the system.
    """
    TAGS = ('essential',)
    STATE_DIMS = ('vertex', 'compt', 'age', 'risk')
    
    _tm_subset = xs.group_dict('edge_weight')
    state = xs.variable(dims=STATE_DIMS, intent='inout', global_name='state',
                        description="The current state of the simulation. "
                        "In the context of a compartmental model, this tensor "
                        "contains the current populations of all compartments "
                        "along every demographic axis (e.g. age, risk, "
                        "geospatial vertex).")
    tm = xs.variable(dims=STATE_DIMS, intent='out', global_name='edge_weight', 
                     description="The transition matrix (`tm` for short) that "
                     "is added to the state at time `t` to get the state for "
                     "the state at the `t + 1` timestep.")
    compt_graph = xs.variable('compt_graph', intent='in', global_name='compt_graph',
                              description="The compartment graph is a networkx "
                              "graph where nodes are compartments and edges "
                              "are allowed transitions between them in the "
                              "compartmental model.")
    stochastic = xs.variable(global_name='stochastic', intent='in', 
                             description="boolean flag that determines " 
                             "whether to sample edge weights stochastically " 
                             "(stochastic=True) or not (deterministic, " 
                             "stochastic=False)")
    seed_state = xs.variable(global_name='seed_state', intent='in',
                             description="integer seed for generating the "
                             "random number generator at every time step.")


    def run_step(self):
        """In particular, we need to ensure that `tm_subset` and `tm` refresh
        at every timestep.
        """
        self._edge_weight_cache = dict()
        self.tm = xr.zeros_like(self.state)
        self.tm_subset = group_dict_by_var(self._tm_subset)
        self.apply_edges()

    def finalize_step(self):
        self.state += self.tm 
        # DEBUG
        assert not self.state.isnull().any()
        # set NaN and near-zero negative values (-1e-8 < x < 0) to zero
        self.state = self.state.where(np.logical_or((self.state >= 0), (self.state < -1e-8)), 0.)
        # DEBUG
        assert not any_negative(self.state, raise_err=True)

    def apply_edges(self) -> None:
        """Iterate over edges in `compt_graph` in ascending order of `priority`.
        Apply each edge to the TM.
        """
        for priority, edges in self.edges_by_priority:
            assert edges, f"no edges with priority={priority}"
            k = 1. if np.isinf(priority) else self.calc_k(*edges)
            self.edge_to_tm(*edges, k=k)

    @property
    def edges_by_priority(self) -> tuple:
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
        assert len(u_set) == 1, f"edges do not share origin node: edges={edges} u_set{u_set}"
        sum_wt = sum(self.edge_weight(u, v) for u, v in edges)
        u_state = self.state.loc[dict(compt=u_set[0])]
        # set k to infinite if denominator is zero
        k = (u_state / sum_wt).fillna(np.Inf)
        # assert np.all(sum_wt), f"u_state={u_state}\nsum_wt={sum_wt}\nk={k}"
        return xr.ufuncs.minimum(k, xr.ones_like(k))

    def edge_to_tm(self, *edges, k=1.) -> None:
        """Applies to the transition matrix (TM) the weight of directed edges
        (u, v) from compartment `u` to compartment `v`. Scale edge weights by
        `k` to ensure that population of origin node `u` is always non-negative
        after edge weight has been applied.
        """
        for u, v in edges:
            weight = k * self.edge_weight(u, v)
            try:
                self.tm.loc[dict(compt=u)] -= weight
                self.tm.loc[dict(compt=v)] += weight
            except ValueError:
                msg = (
                    f"Error while applying weight for edge {(u, v)}. "
                    f"Transition matrix expects matrix with coords:\n "
                    f"{self.tm.loc[dict(compt=u)].coords}\n...but weight "
                    f"has coords:\n {weight.coords}.")
                logging.error(msg)
                raise
            except Exception:
                msg = (
                    f"Unknown error while applying weight for edge {(u, v)}. "
                    f"The edge weight has type '{type(weight)}', and we tried "
                    f"to add it to transition matrix with coordinates\n"
                    f"{self.tm.loc[dict(compt=u)].coords}")
                if isinstance(weight, xr.DataArray):
                    msg = msg + f"\nEdge weight has coordinates\n{weight.coords}"
                logging.error(msg)
                raise
                

    def edge_weight(self, u, v):
        if (u, v) in self._edge_weight_cache:
            return self._edge_weight_cache[u, v]
        else:
            self._edge_weight_cache[u, v] = self.get_edge_weight(u, v)
            return self._edge_weight_cache[u, v]

    def get_edge_weight(self, u, v):
        """Try to find an edge weight for (u, v) in `tm_subset`, then
        in the edge attribute. Default to zero weight if none can be found."""
        key = edge_weight_name(u, v)
        edge_data = self.compt_graph.get_edge_data(u, v, dict())
        if key in self.tm_subset:
            weight = self.tm_subset[key]
        elif 'weight' in edge_data:
            # TODO: accept function and str type attribs
            weight = edge_data['weight']
        else:
            logging.warning(f"could not find a weight for transition from {u} "
                            f"to {v} compartment ({key}). Setting weight to zero.")
            weight = 0.

        # poisson draw if stochastic is on
        if bool(self.stochastic):
            weight = self.stochastic_draw(weight)
        
        # raise a warning if edge appears to be an array, but not a DataArray
        if not isinstance(weight, xr.DataArray) and hasattr(weight, 'shape'):
            logging.error(
                f"Edge weight '{key}' appears to be an array (has 'shape' "
                f"attribute), but is not an xarray.DataArray (is type "
                f"'{type(weight)}'). Array may fail to broadcast with "
                f"other DataArrays. To avoid this, please set '{key}' as a "
                f"float or int if it is a scalar, and an xarray.DataArray if "
                f"it is a tensor/vector.")
        
        # print(f"adjusted weight of edge from {u} to {v} is {weight}")

        # Load Dask arrays into memory
        if hasattr(weight, 'load'):
            weight = weight.load()
            
        return weight

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = get_rng(seed=self.seed_state)
        return self._rng
    
    def stochastic_draw(self, *args, **kwargs):
        return self.poisson(*args, **kwargs)

    def poisson(self, val):
        try:
            arr = self.rng.poisson(val)
        except ValueError as err:
            if "lam value too large" in str(err):
                arr = np.stack([self.poisson(a) for a in val], axis=0)
            else:
                raise
        if isinstance(val, xr.DataArray):
            cp = val.copy()
            cp[{}] = arr
            return cp
        return arr

