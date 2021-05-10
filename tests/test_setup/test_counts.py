import pytest
import pandas as pd
from episimlab.setup.counts import InitCountsFromCensusCSV


@pytest.fixture
def census_df():
    return pd.read_csv('tests/data/2019_zcta_pop_5_age_groups.csv')


class TestInitCountsFromCensusCSV:

    def test_can_initialize(self, census_df, counts_coords):
        """
        """
        inputs = {
            'vertex': counts_coords['vertex'],
            'age_group': counts_coords['age_group'],
            'risk_group': counts_coords['risk_group'],
            'compartment': counts_coords['compartment'],
        }
        proc = InitCountsFromCensusCSV(**inputs)
        proc.initialize()