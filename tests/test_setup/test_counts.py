import pytest
import xarray as xr
import pandas as pd
from episimlab.setup.counts import SetupStateFromCensusCSV


@pytest.fixture
def census_counts_csv():
    return 'tests/data/2019_zcta_pop_5_age_groups.csv'


class TestSetupCountsFromCensusCSV:

    def test_can_initialize(self, census_counts_csv, counts_coords):
        """
        """
        age_coords = counts_coords['age_group']
        # census data uses <5 instead of 0-4
        age_coords[age_coords.index('0-4')] = '<5'
        inputs = {
            # must be zcta that exist in census_counts_csv
            'coords': {
                ('foo', 'vertex'): [75001, 75002, 79312],
                ('foo', 'age'): age_coords,
                ('foo', 'risk'): counts_coords['risk_group'],
                ('foo', 'compt'): counts_coords['compartment'],
            },
            'census_counts_csv': census_counts_csv,
        }
        proc = SetupStateFromCensusCSV(**inputs)
        proc.initialize()
        result = proc.state
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {'vertex', 'age', 'risk', 'compt'}