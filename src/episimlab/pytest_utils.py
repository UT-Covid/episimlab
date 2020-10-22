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

def profiler(flavor='dask', log_dir='./logs', log_stub=None, show_prof=False):
    """Decorates `func` with Dask memory and thread profiling. This function
    returns a decorator, so use like:

    @dask_prof()
    def my_func():
        pass
    """

    assert log_dir is not None
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
            top_stats = snapshot.statistics('filename')
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

            elapsed = time.time() - start
            logging.debug(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    # choose decorator
    if flavor == 'dask':
        decorator = dask_prof
    elif flavor in ('mem', 'ram'):
        decorator = mem_prof

    return decorator

def _get_timestamp():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
