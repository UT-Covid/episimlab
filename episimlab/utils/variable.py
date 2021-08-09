import xsimlab as xs
import xarray as xr


def get_var_dims(process, name) -> tuple:
    """Given process-wrapped class `process`, retrieve the `dims` metadata
    attribute for variable with `name`.
    """
    if not '__xsimlab_cls__' in dir(process):
        raise TypeError(
            f"Expected type 'xsimlab.Process' for arg `process`, received " +
            f"'{type(process)}'"
        )
    var = xs.utils.variables_dict(process).get(name, None)
    if var is None:
        raise AttributeError(f"process '{process}' has no attribute '{name}'")
    return tuple(var.metadata['dims'][0])


def group_dict_by_var(d: dict) -> dict:
    return {k: d[(proc, k)] for (proc, k) in d}


def trim_data_to_coords(data, coords):
    """Be sure to check that order of dims in data is same as order of
    dims in coords.
    """
    assert isinstance(coords, list)
    assert isinstance(data, np.ndarray)
    expected_shape = [len(c) for dim, c in coords]
    # check that N-D is same
    assert len(data.shape) == len(expected_shape)
    # list of slices, one for each dimension
    slices = list()
    for dc, ec in zip(data.shape, expected_shape):
        if ec < dc:
            cut = ec
        else:
            cut = None
        slices.append(slice(None, cut))
    # trim data to expected shape
    return data.__getitem__(slices)
