import xsimlab as xs
import pandas as pd
from .implicit_node import partition_contacts, contact_matrix


@xs.process
class NaiveMigration:

    travel_fp = xs.variable(intent='in')
    contacts_fp = xs.variable(intent='in')

    def initialize(self):
        # Load dataframes
        self.travel = pd.read_csv(self.travel_fp)
        self.contacts = pd.read_csv(self.contacts_fp)
        daily_timesteps = 10

        # Call functions from SEIR_Example
        self.tc_final = partition_contacts(self.travel, self.contacts,
                                           daily_timesteps=daily_timesteps)
        self.phi = contact_matrix(self.tc_final)
