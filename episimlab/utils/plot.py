import os
import logging
import xarray as xr
import xsimlab as xs
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from ..compt_model import edge_weight_name
from typing import Callable


def visualize_compt_graph(
    gph: nx.Graph, path: str = None, mpl_backend: str = None, pos = None,
    default_edge_color: str = 'black', default_node_color: str = 'black',
    node_size: int = 1000, font_size: int = 16, font_weight: str = 'bold',
    font_color: str = 'white', **kwargs):
    """Visualize compartment graph `gph` using matplotlib. Saves figure
    to a `path` if specified, e.g. 'my_compt_graph.svg'.
    """
    
    if mpl_backend is not None:
        # matplotlib.use("Agg")
        matplotlib.use(mpl_backend)
    f = plt.figure()

    # try to get "color" edge attribute if it was set
    edge_color = [edge[2] for edge in gph.edges.data("color", default=default_edge_color)]

    # try to get "color" node attribute if it was set
    node_color = [node[1] for node in gph.nodes.data("color", default=default_node_color)]

    # get edge labels with same naming as compt_model.ComptModel
    edge_labels = {edge: edge_weight_name(*edge) for edge in gph.edges}

    if pos is None:
        pos = nx.drawing.planar_layout(gph)
    ax = f.add_subplot(111)
    drawing = nx.draw_networkx(gph, pos=pos, 
                               ax=ax, edge_color=edge_color, 
                               node_color=node_color, node_size=node_size,
                               font_size=font_size,
                               font_weight=font_weight, font_color=font_color,
                               **kwargs)
    edge_labels = nx.draw_networkx_edge_labels(gph, pos=pos, ax=ax, 
                                               edge_labels=edge_labels)
    if path is not None:
        f.savefig(path)
    return drawing


def xr_plot(data_array, sel=dict(), isel=dict(), timeslice=slice(0, 100),
            sum_over=['risk_group', 'age_group']):
    """Uses DataArray.plot, which builds on mpl
    """
    assert isinstance(data_array, xr.DataArray)
    isel.update({'step': timeslice})
    da = data_array[isel].loc[sel].sum(dim=sum_over)
    return da.plot.line(x='step', aspect=2, size=7)


def plotter(flavor='mpl', log_dir='./logs', log_stub=None, plotter_kwargs=dict()):
    """
    TODO
    WORK IN PROGRESS

    Decorates `func` with function that plots DataArray. This function
    returns a decorator, so use like:

    @plotter()
    def my_func():
        return xr.DataArray()
    """
    raise NotImplementedError()

    assert log_dir is not None
    if log_stub is None:
        log_stub = _get_timestamp()

    def base_fp(name, ext):
        fp = os.path.join(log_dir, f"{log_stub}_{name}.{ext}")
        logging.debug(f"Saving plot to '{fp}'...")
        return fp

    def mpl(func):
        """Decorator
        """
        @wraps(func)
        def with_plot(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, xr.DataArray):
                raise TypeError(f"plotter decorator expected function'{func}' " +
                                f"to return value of type xr.DataArray, " +
                                f"received type '{type(result)}' instead")
            breakpoint()
            plot = xr_plot(result, **plotter_kwargs)
            base_fp(name='da', ext='')
            return result
        return with_plot

    # choose decorator
    if flavor == 'mpl':
        decorator = mpl
    else:
        raise ValueError(f"Could not recognize plotting flavor '{flavor}'")

    return decorator
