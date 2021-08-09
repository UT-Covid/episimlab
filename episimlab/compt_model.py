import numpy as np
import xarray as xr
import xsimlab as xs
import pandas as pd
from .utils import group_dict_by_var


@xs.process
class ComptModel:
    """Applies the compartmental disease model defined in `compt_graph` to the
    current `state` of the system.
    """
    TAGS = ('compt_model', )
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
        """Try to find an edge weight for (u, v) in `tm_subset`, then
        in the edge attribute. Default to zero weight if none can be found."""
        key = self.edge_weight_name(u, v)
        edge_data = self.compt_graph.get_edge_data(u, v, dict())
        if key in self.tm_subset:
            weight = self.tm_subset[key]
        elif 'weight' in edge_data:
            # TODO: accept function and str type attribs
            weight = edge_data['weight']
        else:
            logging.warning(f"could not find a weight for transition from {u} to {v} compartment ({key})")
            weight = 0.
        return weight

    def edge_weight_name(self, u, v) -> str:
        """Attr name when looking for edge weight between nodes `u` and `v`."""
        return f"rate_{u}2{v}"
