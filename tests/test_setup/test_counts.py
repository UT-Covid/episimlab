import pytest
import pandas as pd
from episimlab.setup.counts import InitCountsFromCensusCSV


@pytest.fixture
def census_df():
    return pd.read_csv('tests/data/2019_zcta_pop_5_age_groups.csv')


class TestInitCountsFromCensusCSV:

    def test_can_initialize(self, census_df):
        """
        """
        inputs = dict()
        proc = InitCountsFromCensusCSV(**inputs)
        proc.initialize()