import os
import logging
import xarray as xr
import xsimlab as xs
import numpy as np
from functools import wraps


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
