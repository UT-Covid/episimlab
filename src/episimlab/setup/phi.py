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
    TODO: mimic InitAdjGrpMapping
    """

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    phi_grp1 = xs.index(dims=('phi_grp1'))
    phi_grp2 = xs.index(dims=('phi_grp2'))
    phi_grp_mapping = xs.variable(dims=('age_group', 'risk_group'), static=True, intent='out')
    day_of_week = xs.index(dims=('day_of_week'))

    def initialize(self):
        # self.age_group = ['0-4', '5-17', '18-49', '50-64', '65+']
        # self.risk_group = ['low', 'high']
        encoded_midx = ravel_to_midx(
            # TODO: unhardcode, like adj
            dims=['age_group', 'risk_group'],
            coords=dict(age_group=self.age_group, risk_group=self.risk_group)
        )
        self.phi_grp1 = encoded_midx
        self.phi_grp2 = encoded_midx
        self.day_of_week = np.arange(7)

        self.phi_grp_mapping = self._get_phi_grp_mapping()

    def _get_phi_grp_mapping(self):
        shape = [len(self.age_group), len(self.risk_group)]
        coords = dict(
            age_group=self.age_group,
            risk_group=self.risk_group
        )
        da = xr.DataArray(
            data=self.phi_grp1.reshape(shape),
            dims=('age_group', 'risk_group'),
            coords=coords
        )
        return da


@xs.process
class InitPhi:
    """
    TODO: mimic InitAdj
    """

    phi_grp1 = xs.foreign(InitPhiGrpMapping, 'phi_grp1', intent='in')
    phi_grp2 = xs.foreign(InitPhiGrpMapping, 'phi_grp2', intent='in')
    phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')
    day_of_week = xs.foreign(InitPhiGrpMapping, 'day_of_week', intent='in')
    phi = xs.variable(
        dims=('day_of_week', 'phi_grp1', 'phi_grp2'),
        # dims='phi_grp1',
        static=True,
        intent='out')
    phi_t = xs.variable(dims=('phi_grp1', 'phi_grp2'), intent='out')

    def initialize(self):
        # self.phi = np.zeros(shape=[self.day_of_week.size, self.phi_grp1.size, self.phi_grp2.size], dtype='int32')
        # dims = [self.day_of_week, self.phi_grp1, self.phi_grp2]
        dims = ('day_of_week', 'phi_grp1', 'phi_grp2')
        coords = (('day_of_week', self.day_of_week), ('phi_grp1', self.phi_grp1), ('phi_grp2', self.phi_grp2))
        self.phi = xr.DataArray(
            # TODO
            data=0.75,
            dims=dims,
            coords=coords
        )

    @xs.runtime(args='step')
    def run_step(self, step):
        """
        TODO: use the clock to determine day of week (day_idx)
        """

        # Get the index on `day_of_week`
        day_idx = step % self.day_of_week.size
        self.phi_t = self.phi[day_idx]
        # print(step, self.day_of_week.size, day_idx)
