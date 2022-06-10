import numpy as np
import xsimlab as xs
import xarray as xr


def edge_weight_name(u, v) -> str:
    """Attr name when looking for edge weight between nodes `u` and `v`."""
    return f"rate_{u}2{v}"


def suffixed_dims(da: xr.DataArray, suffix: str, 
                  exclude: list = None) -> dict:
    """Generates a dictionary for suffixing DataArray or DataFrame dimensions.
    Compatible with methods such as `DataFrame.rename` and `DataArray.rename`.
    """
    if exclude is None:
        exclude = list()
    return {k: f"{k}{suffix}" for k in da.dims if k not in exclude}


def unsuffixed_dims(da: xr.DataArray, suffix: str) -> dict:
    """Like `suffixed_dims`, but reverses the renaming transformation."""
    return {v: k for k, v in suffixed_dims(da, suffix).items()}


def get_var_dims(process, name: str) -> tuple:
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
    """Given a dictionary keyed by 2-length tuples, return a dictionary keyed
    only by the second element of the tuple.
    """
    return {k: d[(proc, k)] for (proc, k) in d}


def trim_data_to_coords(data: np.ndarray, coords: list) -> np.ndarray:
    """Given a `data` array, trim the data to match the shape given by list of
    2-length tuples `coords`, formatted like [('dim0', range(10)), 
    ('dim0', range(10)), ...]. Be sure to check that order of dims in data is 
    same as order of dims in coords.
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


def any_negative(val, raise_err=False) -> bool:
    """Checks if there are any negative values in the array. 
    Accepts ndarray and DataArray types.
    """
    err = ''
    tolerance = 0
    if isinstance(val, xr.DataArray):
        any_neg = np.any(val < tolerance)
        if any_neg:
            print(val.nanmin())
            err = f"Found value(s) less than {tolerance} in DataArray: {val.where(val < tolerance, drop=True)}"
    elif isinstance(val, np.ndarray):
        any_neg = np.any(val < tolerance)
        if any_neg:
            print(val.nanmin())
            err = f"Found value(s) less than {tolerance} in ndarray: {np.where(val < tolerance)}"
    elif isinstance(val, xr.Dataset):
        # evaluating the condition returns an xarray -> "any_neg = val.min() < tolerance"
        # is assigned a value and is evaluated as True, even if the only value in that xarray is False
        # a couple of type conversions extract the bool from the xarray for evaluation
        try:
            eval_xr = val.min() < tolerance
            any_neg = eval_xr.to_array().to_numpy().item()
        except ValueError:
            # empty arrays raise value errors... can't be negative if empty...
            any_neg = False
        if any_neg:
            print(val.where(val < tolerance))
            print(f'The minimum value in the xarray dataset is {val.min()}')
            try:
                print(val.where(val < tolerance).estimate)
            except ValueError:
                pass
            err = f"Value of type {type(val)} is less than {tolerance}: {val}"
    else:
        any_neg = val < tolerance
        if any_neg:
            err = f"Value of type {type(val)} is less than {tolerance}: {val}"
    if any_neg and raise_err is True:
        raise ValueError(err)
    else:
        return any_neg


def clip_to_zero(val):
    """numpy.clip that accepts ndarray and DataArray types."""
    if isinstance(val, xr.DataArray):
        return val.clip(min=0.)
    elif isinstance(val, np.ndarray):
        return np.clip(val, 0., np.maximum(val))
    elif isinstance(val, Number):
        return max(val, 0.)


def coerce_to_da(proc, name: str, value, coords: dict = None) -> xr.DataArray:
    """Given a variable with `name` and `value` defined in process `proc`,
    retrieve the variable metadata and use it to coerce the `value` into
    an `xarray.DataArray` with the correct dimensions and coordinates.
    Returns `value` if variable is scalar (zero length dims attribute),
    DataArray otherwise.
    """
    # get dims
    dims = get_var_dims(proc, name)
    if not dims:
        return value
    # get coords
    if coords is None:
        coords = dict()
    return xr.DataArray(data=value, dims=dims, coords={
        dim: coords.get(dim, list()) for dim in dims 
        if dim != 'value' and dim in coords
    })


def fix_coord_dtypes(da: xr.DataArray, max_len: int = None) -> xr.DataArray:
    """Changes coords with object dtype to unicode, e.g. <U5, where 5 would be
    `max_len` in this case. Workaround for missing object_codec for object array.
    """
    for dim in da.coords.keys():
        if da.coords[dim].dtype == 'object':
            len_crd = [len(name) for name in da.coords[dim].values.tolist()]
            if max_len is None:
                new_dtype = f'<U{max(len_crd)}'
            elif any(l > max_len for l in len_crd):
                raise ValueError(f"Tried to set dtype to '<U{max_len}', but "
                                    f"coordinate labels for dim {dim} are "
                                    f"longet than max_len={max_len}.")
            else:
                new_dtype = f'<U{max_len}'
            da.coords[dim] = da.coords[dim].astype(new_dtype)
    return da
