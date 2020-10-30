import xsimlab as xs
import xarray as xr
import attr
from .. import seir, foi, setup, apply_counts_delta, graph

def minimum_viable():
    return xs.Model(dict(
        # Instantiate coords, counts array, default parameters
        init_epi=setup.InitDefaultEpis,
        init_counts=setup.InitDefaultCounts,
        init_coords=setup.InitDefaultCoords,

        # no SEIR engine, these are just placeholders
        foi=foi.brute_force.BaseFOI,
        seir=seir.base.BaseSEIR,

        # Apply all changes made to counts
        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))

def slow_seir():
    model = minimum_viable()
    return model.update_processes(dict(
        # Instantiate phi array
        init_phi=setup.InitPhi,
        init_phi_grp_mapping=setup.InitPhiGrpMapping,
        # Force of infection calculation in python
        foi=foi.brute_force.BruteForceFOI,
        # SEIR engine in python
        seir=seir.brute_force.BruteForceSEIR,
    ))

def cy_seir_w_foi():
    model = minimum_viable()
    return model.update_processes(dict(
        # Instantiate phi array
        init_phi=setup.InitPhi,
        init_phi_grp_mapping=setup.InitPhiGrpMapping,
        # cython SEIR engine, also calculates FOI
        seir=seir.bf_cython_w_foi.BruteForceCythonWFOI,
    ))

def cy_adj():
    model = minimum_viable()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        init_adj=setup.InitToyAdj,
        init_adj_grp_mapping=setup.InitAdjGrpMapping,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=graph.cython.CythonGraph,
    ))

def cy_adj_slow_seir():
    model = slow_seir()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        init_adj=setup.InitToyAdj,
        init_adj_grp_mapping=setup.InitAdjGrpMapping,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=graph.cython.CythonGraph,
    ))

def slow_seir_cy_foi():
    model = slow_seir()
    return model.update_processes(dict(
        foi=foi.bf_cython.BruteForceCythonFOI,
    ))
