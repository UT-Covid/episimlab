import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..coords import InitDefaultCoords
from ...apply_counts_delta import ApplyCountsDelta


@xs.process
class BaseSetupEpi:
    """
    """
    COUNTS_DIMS = ApplyCountsDelta.COUNTS_DIMS

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')
    compartment = xs.foreign(InitDefaultCoords, 'compartment', intent='in')

    @property
    def counts_coords(self):
        return {dim: getattr(self, dim) for dim in self.COUNTS_DIMS}

    def trim_data_to_coords(self, data, coords):
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
