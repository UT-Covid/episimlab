import numpy as np
import numbers


def get_rng(seed: numbers.Number):
    if not isinstance(seed, numbers.Number):
        raise TypeError(f"expected type 'numbers.Number' for arg `seed`, " +
                        f"received type '{type(seed)}'")
    return np.random.default_rng(seed=seed)
