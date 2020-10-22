import xsimlab as xs
import xarray as xr
import attr
from .. import seir, setup, apply_counts_delta, graph

def slow_seir():
    return xs.Model(dict(
        init_epi=setup.InitDefaultEpis,
        init_counts=setup.InitDefaultCounts,
        init_coords=setup.InitDefaultCoords,
        init_phi=setup.InitPhi,
        init_phi_grp_mapping=setup.InitPhiGrpMapping,
        foi=seir.foi.BruteForceFOI,
        seir=seir.seir.BruteForceSEIR,
        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))

def cy_adj_slow_seir():
    return xs.Model(dict(
        init_epi=setup.InitDefaultEpis,
        init_counts=setup.InitDefaultCounts,
        init_coords=setup.InitDefaultCoords,
        init_adj=setup.InitToyAdj,
        init_adj_grp_mapping=setup.InitAdjGrpMapping,
        travel=graph.cython.CythonGraph,

        # WITH slow SEIR
        init_phi=setup.InitPhi,
        init_phi_grp_mapping=setup.InitPhiGrpMapping,
        foi=seir.foi.BruteForceFOI,
        seir=seir.seir.BruteForceSEIR,

        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))

def cy_adj():
    return xs.Model(dict(
        init_epi=setup.InitDefaultEpis,
        init_counts=setup.InitDefaultCounts,
        init_coords=setup.InitDefaultCoords,
        init_adj=setup.InitToyAdj,
        init_adj_grp_mapping=setup.InitAdjGrpMapping,
        travel=graph.cython.CythonGraph,

        # without SEIR
        foi=seir.foi.BaseFOI,
        seir=seir.seir.BaseSEIR,

        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))
