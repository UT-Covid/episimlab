import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ..seir import base
from ..foi.base import BaseFOI
from ..setup.coords import InitDefaultCoords
from ..seir.base import BaseSEIR


@xs.process
class SeedEntropy:
    """Sets seed entropy for testing purposes
    """
    seed_entropy = xs.variable(static=True, intent='out')

    def initialize(self):
        self.seed_entropy = 12345


@xs.process
class SeedGenerator:
    """
    """
    seed_entropy = xs.variable(static=True, intent='in')
    seed_state = xs.foreign(BaseSEIR, 'seed_state', intent='out')

    def initialize(self):
        # instantiate a SeedSequence
        self.seed_seq = np.random.SeedSequence(entropy=(self.seed_entropy))

    def spawn_next(self, seed_seq):
        # spawn a child SeedSequence
        child_ss_lst = seed_seq.spawn(1)
        assert isinstance(child_ss_lst, list), type(child_ss_lst)
        child_ss = child_ss_lst[0]
        assert isinstance(child_ss, np.random.bit_generator.SeedSequence), \
            type(child_ss)

        # get the state of this child SeedSequence
        child_state_arr = child_ss.generate_state(1)
        assert isinstance(child_state_arr, np.ndarray), type(child_state_arr)
        return child_state_arr[0]

    def run_step(self):
        self.seed_state = self.spawn_next(self.seed_seq)
