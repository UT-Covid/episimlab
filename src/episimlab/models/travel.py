import xsimlab as xs
import xarray as xr
import attr

from ..setup import adj
from ..network import cython_explicit_travel
from .basic import minimum_viable


def cy_adj():
    model = minimum_viable()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        setup_adj=adj.InitToyAdj,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython_explicit_travel.CythonExplicitTravel,
    ))


def cy_seir_cy_foi_cy_adj():
    model = cy_seir_cy_foi()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        setup_adj=adj.InitToyAdj,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython_explicit_travel.CythonExplicitTravel,
    ))
