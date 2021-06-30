import os
import attr
import math
from pprint import pprint as pp
from glob import glob
import numpy as np
import pandas as pd


@attr.s
class ScalingExp:
    """One scaling experiment."""

    ref_census_csv = attr.ib(type=str)
    ref_travel_csv = attr.ib(type=str)
    ref_contacts_csv = attr.ib(type=str)
    pop_factor = attr.ib(type=int, default=1)
    zip_factor = attr.ib(type=int, default=1)

    @property
    def census(self, force_refresh=False) -> pd.DataFrame:
        if hasattr(self, '_census') and not force_refresh:
            return self._census
        else:
            return self.parse_census()

    @property
    def census(self, force_refresh=False) -> pd.DataFrame:
        if hasattr(self, '_census') and not force_refresh:
            return self._census
        else:
            return self.parse_census()

    @property
    def census(self, force_refresh=False) -> pd.DataFrame:
        if hasattr(self, '_census') and not force_refresh:
            return self._census
        else:
            return self.parse_census()

    def parse_census(self) -> pd.DataFrame:
        df = pd.read_csv(self.ref_census_csv, usecols=schemas['census']['usecols'])
        assert not df.isna().any().any(), ('found null values in df', df.isna().any())
        # df.rename(columns={'GEOID': 'vertex', 'age_bin': 'age_group'}, inplace=True)
        # df.set_index(['vertex', 'age_group'], inplace=True)
        # filter to zcta that we want to model in the simulation (vertex coords)
        self._census = df
        return df

    def parse_contacts(self) -> pd.DataFrame:
        self._contacts = pd.read_csv(self.ref_contacts_csv, usecols=schemas['contacts']['usecols'])
        return self._contacts

    def parse_travel(self) -> pd.DataFrame:
        self._travel = pd.read_csv(self.ref_travel_csv, usecols=schemas['travel']['usecols'])
        return self._travel


def main():
    exp = ScalingExp(
        ref_census_csv='../data/lccf/census_pop1_rows1.csv',
        ref_travel_csv='../data/lccf/travel_pop1_rows1.csv',
        ref_contacts_csv='../data/lccf/contacts_pop1_rows1.csv',
    )
    print(exp)


if __name__ == '__main__':
    main()
