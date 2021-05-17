import pandas as pd
import xarray as xr
import xsimlab as xs
import numpy as np
from itertools import product
from ..setup.coords import InitDefaultCoords
from . import partition


def load_travel_df(travel_fp):
    tdf = pd.read_csv(travel_fp, dtype={'date': str})
    try:
        tdf = tdf.rename(columns={'age_src': 'age'})
    except KeyError:
        pass

    return tdf


@xs.process
class TravelManager:

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')
    age_group = xs.foreign(InitDefaultCoords, 'age_group')
    # todo: TravelManager does nothing with age_group other than pass to Partition -- better organization?

    def initialize(self):

        self.baseline_contact_df = pd.read_csv(self.contacts_fp)
        self.travel_df = load_travel_df(self.travel_fp)

    @xs.runtime
    def run_step(self, time_step_date):

        travel_current_date = self.subset_date(time_step_date)
        part = partition.Partition(
            travel_df=travel_current_date,
            baseline_contact_df=self.baseline_contact_df,
            age_group=self.age_group
        )
        part.initialize()

        # other SEIR processes can access the contact matrix without direct interface with the Partition object
        self.contact_xr = part.contact_xr

    def subset_date(self, time_step_date):

        travel_current_date = self.travel_df[self.travel_df['date'] == time_step_date]
        try:
            assert travel_current_date.empty == False
        except AssertionError as e:
            e.args += (('No travel data for date {}.'.format(time_step_date), ))
            raise

        return travel_current_date
