import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd
import itertools

from ..setup.coords import InitDefaultCoords
from ..utils import ravel_to_midx, unravel_encoded_midx


@xs.process
class InitAdjGrpMapping:
    """
    """
    MAP_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')
    compartment = xs.foreign(InitDefaultCoords, 'compartment', intent='in')

    adj_grp1 = xs.index(dims=('adj_grp1'))
    adj_grp2 = xs.index(dims=('adj_grp2'))
    adj_grp_mapping = xs.variable(dims=MAP_DIMS, static=True, intent='out')

    def initialize(self):
        self.MAP_COORDS = {
            k: getattr(self, k) for k in self.MAP_DIMS
        }

        encoded_midx = ravel_to_midx(dims=self.MAP_DIMS, coords=self.MAP_COORDS)
        self.adj_grp1 = encoded_midx
        self.adj_grp2 = encoded_midx

        self.adj_grp_mapping = self._get_adj_grp_mapping()

    def _get_adj_grp_mapping(self):
        shape = [len(coord) for coord in self.MAP_COORDS.values()]
        da = xr.DataArray(
            data=self.adj_grp1.reshape(shape),
            dims=self.MAP_DIMS,
            coords=self.MAP_COORDS
        )
        return da


@xs.process
class InitToyAdj:
    """TODO: Separate the adj_t slicing and adj
    instantiation into separate processes
    """
    ADJ_DIMS = ('day_of_week', 'vertex1', 'vertex2',
                'age_group', 'risk_group', 'compartment')

    age_group = xs.foreign(InitAdjGrpMapping, 'age_group', intent='in')
    risk_group = xs.foreign(InitAdjGrpMapping, 'risk_group', intent='in')
    vertex = xs.foreign(InitAdjGrpMapping, 'vertex', intent='in')
    compartment = xs.foreign(InitAdjGrpMapping, 'compartment', intent='in')
    # TODO: set these coords separately
    day_of_week = xs.index(dims=('day_of_week'))

    adj = xs.variable(
        dims=ADJ_DIMS,
        static=True,
        intent='out')
    adj_t = xs.variable(dims=ADJ_DIMS[1:], intent='out')

    def initialize(self):
        # Set dims for vertex1 and vertex2 same as for dim vertex
        self.vertex1 = self.vertex
        self.vertex2 = self.vertex

        # Set day of week coords
        self.day_of_week = np.arange(7)

        self.ADJ_COORDS = {k: getattr(self, k) for k in self.ADJ_DIMS}
        self.adj = xr.DataArray(
            data=0.,
            dims=self.ADJ_DIMS,
            coords=self.ADJ_COORDS
        )

    @xs.runtime(args='step')
    def run_step(self, step):
        """
        TODO: this assumes that the first day is Monday...
        """
        # Get the index on `day_of_week`
        day_idx = step % 7
        self.adj_t = self.adj[day_idx]
        # print(step, self.day_of_week.size, day_idx)
