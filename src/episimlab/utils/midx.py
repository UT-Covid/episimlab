import numpy as np
import pandas as pd
from math import isclose
from collections.abc import Iterable


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
