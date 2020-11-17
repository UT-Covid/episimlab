import xsimlab as xs
import xarray as xr
import attr

from ..setup import seed, sto, epi, counts, coords, adj, phi
from ..foi import (
    base as base_foi,
    brute_force as bf_foi,
    bf_cython as bf_cython_foi
)
from ..seir import (
    base as base_seir,
    brute_force as bf_seir,
    bf_cython as bf_cython_seir,
    bf_cython_w_foi
)
from .. import apply_counts_delta
from ..graph import cython


def minimum_viable():
    return xs.Model(dict(
        # Random number generator
        rng=seed.SeedGenerator,
        sto=sto.InitStochasticFromToggle,

        # Instantiate coords, counts array, default parameters
        init_epi=epi.InitDefaultEpis,
        init_counts=counts.InitDefaultCounts,
        init_coords=coords.InitDefaultCoords,

        # no SEIR engine, these are just placeholders
        foi=bf_foi.BaseFOI,
        seir=base_seir.BaseSEIR,

        # Apply all changes made to counts
        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))

def slow_seir():
    model = minimum_viable()
    return model.update_processes(dict(
        # Instantiate phi array
        init_phi=phi.InitPhi,
        init_phi_grp_mapping=phi.InitPhiGrpMapping,
        # Force of infection calculation in python
        foi=bf_foi.BruteForceFOI,
        # SEIR engine in python
        seir=bf_seir.BruteForceSEIR,
    ))

def cy_seir_w_foi():
    model = minimum_viable()
    return model.update_processes(dict(
        # Instantiate phi array
        init_phi=phi.InitPhi,
        init_phi_grp_mapping=phi.InitPhiGrpMapping,
        # cython SEIR engine, also calculates FOI
        seir=bf_cython_w_foi.BruteForceCythonWFOI,
    ))

def cy_adj():
    model = minimum_viable()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        init_adj=adj.InitToyAdj,
        init_adj_grp_mapping=adj.InitAdjGrpMapping,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython.CythonGraph,
    ))

def cy_adj_slow_seir():
    model = slow_seir()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        init_adj=adj.InitToyAdj,
        init_adj_grp_mapping=adj.InitAdjGrpMapping,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython.CythonGraph,
    ))

def slow_seir_cy_foi():
    model = slow_seir()
    return model.update_processes(dict(
        foi=bf_cython_foi.BruteForceCythonFOI,
    ))


def cy_seir_cy_foi():
    model = slow_seir()
    return model.update_processes(dict(
        foi=bf_cython_foi.BruteForceCythonFOI,
        seir=bf_cython_seir.BruteForceCython
    ))
