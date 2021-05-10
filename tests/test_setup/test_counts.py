import pytest
import pandas as pd
from episimlab.setup.counts import InitCountsFromCensusCSV


@pytest.fixture
def census_df():
    return pd.read_csv('tests/data/total_pop_zcta_2019.csv')


class TestInitCountsFromCensusCSV:

    def test_can_initialize(self, census_df):
        """
        """
        inputs = dict()
        proc = InitCountsFromCensusCSV(**inputs)
        proc.initialize()