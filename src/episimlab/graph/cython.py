import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from ..setup.coords import InitDefaultCoords
from ..setup.adj import InitAdjGrpMapping, InitToyAdj
from . import cy_engine

@xs.process
class CythonGraph:
    """Calculate change in `counts` due to travel between nodes.
    """

    COUNTS_DIMS = ('vertex', 'age_group', 'risk_group', 'compartment')

    age_group = xs.foreign(InitDefaultCoords, 'age_group', intent='in')
    risk_group = xs.foreign(InitDefaultCoords, 'risk_group', intent='in')
    vertex = xs.foreign(InitDefaultCoords, 'vertex', intent='in')
    compartment = xs.foreign(InitDefaultCoords, 'compartment', intent='in')

    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')
    counts_delta_gph = xs.variable(
        groups=['counts_delta'],
        dims=COUNTS_DIMS,
        static=False,
        intent='out'
    )

    adj_t = xs.foreign(InitToyAdj, 'adj_t', intent='in')
    adj_grp_mapping = xs.foreign(InitAdjGrpMapping, 'adj_grp_mapping', intent='in')

    def run_step(self):
        """
        """
        assert isinstance(self.adj_t, xr.DataArray)
        assert isinstance(self.counts, xr.DataArray)
        assert isinstance(self.adj_grp_mapping, xr.DataArray)

        self.counts_delta_gph = cy_engine.graph_high_gran(
            self.counts.values, self.adj_t.values, self.adj_grp_mapping.values)

