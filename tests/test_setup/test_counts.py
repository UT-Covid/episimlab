import pytest
import pandas as pd
from episimlab.setup.counts import InitCountsFromCensusCSV


@pytest.fixture
def census_counts_csv():
    return 'tests/data/2019_zcta_pop_5_age_groups.csv'


class TestInitCountsFromCensusCSV:

    def test_can_initialize(self, census_counts_csv, counts_coords):
        """
        """
        inputs = {
            'vertex': counts_coords['vertex'],
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'compartment': counts_coords['compartment'],
            'census_counts_csv': census_counts_csv,
        }
        proc = InitCountsFromCensusCSV(**inputs)
        proc.initialize()
        result = proc.counts
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {'vertex', 'age_group', 'risk_group', 'compartment'}