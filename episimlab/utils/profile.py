import os
import xarray as xr
import xsimlab as xs
import numpy as np
from dask.diagnostics import ResourceProfiler, Profiler, ProgressBar
from functools import wraps
import tracemalloc
import datetime
import time
import logging


def profiler(flavor='wall_clock', log_dir='./', log_stub=None, show_prof=False,
             cumulative=False):
    """Decorates `func` with Dask memory and thread profiling. This function
    returns a decorator, so use like:

    @profiler()
    def my_func():
        pass
    """

    assert os.path.isdir(log_dir)
    if log_stub is None:
        log_stub = _get_timestamp()

    def base_fp(name, ext):
        fp = os.path.join(log_dir, f"{log_stub}_{name}.{ext}")
        logging.debug(f"Saving profiling report to '{fp}'...")
        return fp

    def dask_prof(func):
        """Decorator
        """
        @wraps(func)
        def with_prof(*args, **kwargs):
            start = time.time()
            with ResourceProfiler(dt=0.25) as rprof, Profiler() as tprof:
                result = func(*args, **kwargs)
            rprof.visualize(file_path=base_fp(name='rprof', ext='html'),
                            show=show_prof, save=True)
            tprof.visualize(file_path=base_fp(name='tprof', ext='html'),
                            show=show_prof, save=True)
            elapsed = time.time() - start
            logging.debug(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    def mem_prof(func):
        """Decorator
        """
        @wraps(func)
        def with_prof(*args, **kwargs):
            start = time.time()
            tracemalloc.start()
            result = func(*args, **kwargs)

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('filename', cumulative=cumulative)
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

            elapsed = time.time() - start
            logging.debug(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    def wall_clock(func):
        """Decorator
        """
        @wraps(func)
        def with_prof(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.debug(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    # choose decorator
    if flavor == 'dask':
        decorator = dask_prof
    elif flavor in ('mem', 'ram'):
        decorator = mem_prof
    elif flavor in ('timer', 'wall_clock'):
        decorator = wall_clock

    return decorator


def _get_timestamp():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')


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
