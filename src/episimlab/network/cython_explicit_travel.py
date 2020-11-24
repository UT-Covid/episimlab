import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..seir.base import BaseSEIR
from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from ..setup.adj import InitAdjGrpMapping, InitToyAdj
from .cython_explicit_travel_engine import graph_high_gran

@xs.process
class CythonExplicitTravel:
    """Calculate change in `counts` due to travel between nodes.
    """

    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')
    compartment = xs.foreign(InitDefaultCoords, 'compartment', intent='in')

    # TODO: switch to globals
    stochastic = xs.foreign(BaseSEIR, 'stochastic', intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='in')

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    counts_delta_gph = xs.variable(
        groups=['counts_delta'],
        dims=COUNTS_DIMS,
        static=False,
        intent='out'
    )

    adj_t = xs.foreign(InitToyAdj, 'adj_t', intent='in')
    adj_grp_mapping = xs.foreign(InitAdjGrpMapping, 'adj_grp_mapping', intent='in')

    def initialize(self):
        # Ensure that coords for counts and adj_grp_mapping are the same
        for k, counts_coords in self.counts.coords.items():
            xr.testing.assert_equal(counts_coords, self.adj_grp_mapping.coords[k])

        # Same for dims, also must be in same order
        for i, expected in enumerate(self.COUNTS_DIMS):
            assert self.counts.dims[i] == expected, \
                f"expected '{expected}' but got '{self.counts.dims[i]}'"
            assert self.adj_grp_mapping.dims[i] == expected, \
                f"expected '{expected}' but got '{self.adj_grp_mapping.dims[i]}'"

        # xr.testing.assert_equal(self.counts.coords, self.adj_grp_mapping.coords)

    def run_step(self):
        """
        """
        assert isinstance(self.adj_t, xr.DataArray)
        assert isinstance(self.counts, xr.DataArray)
        assert isinstance(self.adj_grp_mapping, xr.DataArray)

        self.counts_delta_gph_arr = graph_high_gran(
            counts=self.counts.values,
            adj_t=self.adj_t.values,
            adj_grp_mapping=self.adj_grp_mapping.values,
            stochastic=self.stochastic,
            int_seed=self.seed_state
        )

    def finalize_step(self):
        self.counts_delta_gph = xr.DataArray(
            data=self.counts_delta_gph_arr,
            dims=self.counts.dims,
            coords=self.counts.coords
        )
