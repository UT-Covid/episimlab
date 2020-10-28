import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd
import itertools
from collections.abc import Iterable

from ..setup.coords import InitDefaultCoords


@xs.process
class InitPhiGrpMapping:
    """TODO: handle the coords dynamically as a `group`
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
        encoded_midx = get_encoded_midx(coords=dict(age_group=self.age_group, risk_group=self.risk_group))
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
            coords=dict(
                age_group=self.age_group,
                risk_group=self.risk_group
            )
        )
        return da


@xs.process
class InitPhi:
    """
    """

    phi_grp1 = xs.foreign(InitPhiGrpMapping, 'phi_grp1', intent='in')
    phi_grp2 = xs.foreign(InitPhiGrpMapping, 'phi_grp2', intent='in')
    phi_grp_mapping = xs.foreign(InitPhiGrpMapping, 'phi_grp_mapping', intent='in')
    age_group = xs.foreign(InitPhiGrpMapping, 'age_group', intent='in')
    risk_group = xs.foreign(InitPhiGrpMapping, 'risk_group', intent='in')
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
        self.phi = xr.DataArray(data=0., dims=dims, coords=coords)

    @xs.runtime(args='step')
    def run_step(self, step):

        # Get the index on `day_of_week`
        day_idx = step % self.day_of_week.size
        self.phi_t = self.phi[day_idx]
        # print(step, self.day_of_week.size, day_idx)

    def _toy_finalize_step(self):
        """Toy behavior of how the SEIR model would access this array

        TODO: a version of this with matrix multiplication
        """

        # Iterate over every pair of age-risk pairs
        for a1, r1, a2, r2 in itertools.product(*[self.age_group, self.risk_group] * 2):
            # print(combo)

            # Get the indices in phi_grp1/phi_grp2
            i = self.phi_grp_mapping.loc[(a1, r1)].values
            j = self.phi_grp_mapping.loc[(a2, r2)].values
            # print(i, j)

            # ...then index the symmetrical array using
            # the derived i j indices
            self.phi_t[i, j] += 1

        # Validate that this encoded index actually works
        # print(self._validate_midx())

    def _validate_midx(self):
        # Validate that this is actually the correct index in phi_grp1
        return ravel_encoded_midx(
            midx=self.phi_grp1,
            coords=dict(
                age_group=self.age_group,
                risk_group=self.risk_group
            )
        )


def get_encoded_midx(coords):
    """TODO: pass dims so we dont rely on coords.keys() order
    """
    # Type checking
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

    # Generate pandas MultiIndex
    shape = [_c.size for _c in c.values()]
    midx = pd.MultiIndex.from_product(c.values(), names=c.keys())
    return np.ravel_multi_index(midx.codes, shape)

# USAGE
# encoded_midx = get_encoded_midx(dict(
    # ag=np.array(['0-4', '5-17', '18-49', '50-64', '65+']),
    # rg=np.array(['low', 'high'])
# ))
# encoded_midx

def ravel_encoded_midx(midx, coords):
    """TODO: pass dims so we dont rely on coords.keys() order
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
    shape = [_c.size for _c in c.values()]
    indices = np.unravel_index(midx, shape)
    arrays = [c[dim][index] for dim, index in zip(c.keys(), indices)]
    return pd.MultiIndex.from_arrays(arrays)


# USAGE
# decoded_midx = ravel_encoded_midx(
    # midx=encoded_midx,
    # coords=dict(
        # ag=np.array(['0-4', '5-17', '18-49', '50-64', '65+']),
        # rg=np.array(['low', 'high'])
    # )
# )
# decoded_midx
