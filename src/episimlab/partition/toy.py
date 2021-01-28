import xsimlab as xs
import pandas as pd
from .implicit_node import partition_contacts


@xs.process
class NaiveMigration:

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')

    def initialize(self):
        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)

        # raise NotImplementedError()
