import xsimlab as xs
import xarray as xr
import numpy as np
import pandas as pd
import itertools
from collections.abc import Iterable

from ..setup.coords import InitDefaultCoords


@xs.process
class InitAdjGrpMapping:
    """TODO: handle the coords dynamically as a `group`
    """

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    adj_grp1 = xs.index(dims=('adj_grp1'))
    adj_grp2 = xs.index(dims=('adj_grp2'))
    adj_grp_mapping = xs.variable(dims=('age_group', 'risk_group'), static=True, intent='out')
    day_of_week = xs.index(dims=('day_of_week'))

    def initialize(self):
        encoded_midx = get_encoded_midx(
            coords=dict(age_group=self.age_group, risk_group=self.risk_group))
        self.adj_grp1 = encoded_midx
        self.adj_grp2 = encoded_midx
        self.day_of_week = np.arange(7)

        self.adj_grp_mapping = self._get_adj_grp_mapping()

    def _get_adj_grp_mapping(self):
        shape = [len(self.age_group), len(self.risk_group)]
        coords = dict(
            age_group=self.age_group,
            risk_group=self.risk_group
        )
        da = xr.DataArray(
            data=self.adj_grp1.reshape(shape),
            dims=('age_group', 'risk_group'),
            coords=dict(
                age_group=self.age_group,
                risk_group=self.risk_group
            )
        )
        return da


@xs.process
class InitToyAdj:
    """
    """

    adj_grp1 = xs.foreign(InitAdjGrpMapping, 'adj_grp1', intent='in')
    adj_grp2 = xs.foreign(InitAdjGrpMapping, 'adj_grp2', intent='in')
    adj_grp_mapping = xs.foreign(InitAdjGrpMapping, 'adj_grp_mapping', intent='in')
    age_group = xs.foreign(InitAdjGrpMapping, 'age_group', intent='in')
    risk_group = xs.foreign(InitAdjGrpMapping, 'risk_group', intent='in')
    day_of_week = xs.foreign(InitAdjGrpMapping, 'day_of_week', intent='in')
    adj = xs.variable(
        dims=('day_of_week', 'adj_grp1', 'adj_grp2'),
        # dims='adj_grp1',
        static=True,
        intent='out')
    adj_t = xs.variable(dims=('adj_grp1', 'adj_grp2'), intent='out')

    def initialize(self):
        # self.adj = np.zeros(shape=[self.day_of_week.size, self.adj_grp1.size, self.adj_grp2.size], dtype='int32')
        # dims = [self.day_of_week, self.adj_grp1, self.adj_grp2]
        dims = ('day_of_week', 'adj_grp1', 'adj_grp2')
        coords = (('day_of_week', self.day_of_week), ('adj_grp1', self.adj_grp1), ('adj_grp2', self.adj_grp2))
        self.adj = xr.DataArray(data=0, dims=dims, coords=coords)

    @xs.runtime(args='step')
    def run_step(self, step):

        # Get the index on `day_of_week`
        day_idx = step % self.day_of_week.size
        self.adj_t = self.adj[day_idx]
        # print(step, self.day_of_week.size, day_idx)

    def _toy_finalize_step(self):
        """Toy behavior of how the graph model would access this array

        TODO: a version of this with matrix multiplication
        """

        # Iterate over every pair of age-risk pairs
        for a1, r1, a2, r2 in itertools.product(*[self.age_group, self.risk_group] * 2):
            # print(combo)

            # Get the indices in adj_grp1/adj_grp2
            i = self.adj_grp_mapping.loc[(a1, r1)].values
            j = self.adj_grp_mapping.loc[(a2, r2)].values
            # print(i, j)

            # ...then index the symmetrical array using
            # the derived i j indices
            self.adj_t[i, j] += 1

        # Validate that this encoded index actually works
        # print(self._validate_midx())

    def _validate_midx(self):
        # Validate that this is actually the correct index in adj_grp1
        return ravel_encoded_midx(
            midx=self.adj_grp1,
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
