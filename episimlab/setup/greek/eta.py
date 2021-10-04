import xsimlab as xs
import xarray as xr
import numpy as np
import logging


@xs.process
class SetupDefaultEta:
    """Return a default value for eta."""
    eta = xs.global_ref('eta', intent='out')

    def initialize(self):
        self.eta = self.get_eta()

    def get_eta(self):
        return 0.169492


@xs.process
class SetupEtaFromAsympRate(SetupDefaultEta):
    """Given a static scalar time until hospitalization, calculate eta."""
    t_onset_to_Ih = xs.variable(dims=(), static=True, intent='in')

    def get_eta(self):
        return 1 / self.t_onset_to_Ih
