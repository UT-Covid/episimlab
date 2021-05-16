import pytest
import os
import pandas as pd
import numpy as np
import yaml
import xarray as xr
import xsimlab as xs
from itertools import product

from episimlab.partition import partition, travel_management as tm
from .test_partition import to_phi_da

@pytest.fixture
def travel_mgmt_data():
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    return {
        'contacts_fp': os.path.join(base_dir, 'contacts8.csv'),
        'travel_fp': os.path.join(base_dir, 'travel8.csv'),
        'age_group': ['young', 'old']
    }

@pytest.fixture
def results():
    base_dir = os.path.join('tests', 'data', 'partition_capture')
    return {'contact_xr': os.path.join(base_dir, 'phi8.npy')}

@pytest.fixture
def test_dates_pass():

    return '2020-03-11'

@pytest.fixture
def test_dates_fail():

    return '2050-01-01'


class TestTravelManager:
    """
    Check that travel manager can pass dataframe subsets to Partition
    """

    # data frame contains only one date; subset_date() should return the df unchanged
    def test_date_slice(self, travel_mgmt_data, test_dates_pass):

        travel = tm.TravelManager(**travel_mgmt_data)
        travel.initialize()
        date_df = travel.subset_date(test_dates_pass)
        pd.testing.assert_frame_equal(travel.travel_df, date_df)

    def test_missing_date(self, travel_mgmt_data, test_dates_fail):

        travel = tm.TravelManager(**travel_mgmt_data)
        travel.initialize()
        with pytest.raises(Exception):
            _ = travel.subset_date(test_dates_fail)

    @pytest.mark.xfail
    def test_run_step(self, travel_mgmt_data, test_dates_pass, results, to_phi_da):

        travel = tm.TravelManager(**travel_mgmt_data)
        travel.initialize()
        travel.run_step(test_dates_pass)

        # construct a DataArray from legacy phi
        phi = to_phi_da(results['contact_xr'])

        # sort each coordinate
        # this just changes assert_allclose to be agnostic to order of coords
        def sort_coords(da):
            for dim in da.dims:
                da = da.sortby(dim)
            return da

        # rename test output to have legacy coordinate names
        test_phi_xr = travel.contact_xr.rename(
            {'age_i': 'age_group1', 'age_j': 'age_group2', 'vertex_i': 'vertex1', 'vertex_j': 'vertex2'}
        )
        xr.testing.assert_allclose(sort_coords(test_phi_xr), sort_coords(phi))
