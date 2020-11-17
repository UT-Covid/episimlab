import numpy as np


def foo(ns):
    return ns / np.timedelta64(1, 'D').astype('timedelta64[ns]')
