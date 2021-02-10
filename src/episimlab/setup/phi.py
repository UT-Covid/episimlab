import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd
import itertools

from ..setup.coords import InitDefaultCoords
from ..utils import ravel_to_midx, unravel_encoded_midx


@xs.process
class InitPhiGrpMapping:
    """
    """
    DIMS = ('vertex', 'age_group', 'risk_group')

    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    vertex = xs.global_ref('vertex')
    phi_grp1 = xs.index(dims=('phi_grp1'))
    phi_grp2 = xs.index(dims=('phi_grp2'))
    phi_grp_mapping = xs.variable(dims=DIMS, static=True, intent='out')

    def initialize(self):
        self.COORDS = {k: getattr(self, k) for k in self.DIMS}
        encoded_midx = ravel_to_midx(dims=self.DIMS, coords=self.COORDS)
        self.phi_grp1 = encoded_midx
        self.phi_grp2 = encoded_midx
        self.phi_grp_mapping = self._get_phi_grp_mapping()

    def _get_phi_grp_mapping(self):
        shape = [len(coord) for coord in self.COORDS.values()]
        return xr.DataArray(
            data=self.phi_grp1.reshape(shape),
            dims=self.DIMS,
            coords=self.COORDS
        )


@xs.process
class InitPhi:
    """
    """
    DIMS = ('phi_grp1', 'phi_grp2')

    phi_grp1 = xs.foreign(InitPhiGrpMapping, 'phi_grp1', intent='in')
    phi_grp2 = xs.foreign(InitPhiGrpMapping, 'phi_grp2', intent='in')
    phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')
    phi = xs.variable(dims=DIMS, static=True, intent='out')
    phi_t = xs.variable(dims=DIMS, intent='out', global_name='phi_t')

    def initialize(self):
        # coords = ((dim, getattr(self, dim)) for dim in self.DIMS)
        self.COORDS = {k: getattr(self, k) for k in self.DIMS}
        # TODO
        data = 0.75
        self.phi = xr.DataArray(data=data, dims=self.DIMS, coords=self.COORDS)
        self.phi_t = self.phi

    @xs.runtime(args='step')
    def run_step(self, step):
        pass
