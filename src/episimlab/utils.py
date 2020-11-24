import numpy as np
import pandas as pd
from math import isclose
from collections.abc import Iterable
from .cy_utils.cy_utils import discrete_time_approx_wrapper as cy_dta


def discrete_time_approx(rate, timestep):
    """
    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    # if rate >= 1:
        # return np.nan
    # elif timestep == 0:
        # return np.nan

    val = 1. - (1. - rate)**(1. / timestep)
    cy_val = cy_dta(rate, timestep)
    assert isclose(val, cy_val, rel_tol=1e-7), (val, cy_val)
    return val


def ravel_to_midx(dims, coords):
    """
    # USAGE
    # encoded_midx = ravel_to_midx(dict(
        # ag=np.array(['0-4', '5-17', '18-49', '50-64', '65+']),
        # rg=np.array(['low', 'high'])
    # ))
    # encoded_midx
    """
    # Type checking
    assert isinstance(coords, dict)
    c = coords.copy()
    # Convert elements to ndarray
    for k, v in c.items():
        # Since we're using the `size` attr...
        if isinstance(v, np.ndarray):
            pass
        elif isinstance(v, Iterable):
            c[k] = np.array(v)
        else:
            raise TypeError()

    # Generate pandas MultiIndex
    values = [c[dim] for dim in dims]
    shape = [v.size for v in values]
    midx = pd.MultiIndex.from_product(values, names=dims)
    return np.ravel_multi_index(midx.codes, shape)


def unravel_encoded_midx(midx, dims, coords):
    """
    # USAGE
    # decoded_midx = unravel_encoded_midx(
        # midx=encoded_midx,
        # coords=dict(
            # ag=np.array(['0-4', '5-17', '18-49', '50-64', '65+']),
            # rg=np.array(['low', 'high'])
        # )
    # )
    # decoded_midx
    """
    # Type checking
    assert isinstance(midx, np.ndarray)
    assert isinstance(coords, dict)
    c = coords.copy()
    for k, v in c.items():
        # Since we're using the `size` attr...
        if isinstance(v, np.ndarray):
            pass
        elif isinstance(v, Iterable):
            c[k] = np.array(v)
        else:
            raise TypeError()

    # Decode to a MultiIndex
    values = [c[dim] for dim in dims]
    shape = [v.size for v in values]
    indices = np.unravel_index(midx, shape)
    arrays = [c[dim][index] for dim, index in zip(dims, indices)]
    return pd.MultiIndex.from_arrays(arrays)
