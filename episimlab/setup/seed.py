import xsimlab as xs
import xarray as xr
import numpy as np
import logging


@xs.process
class SeedGenerator:
    """Handles the `seed_state` variable, which is a pseudo-randomly generated
    integer seed that changes at every step of the simulation. Generation is
    handled by a numpy SeedSequence, which itself is seeded by an input integer
    seed `seed_entropy`.
    """
    seed_entropy = xs.variable(static=True, intent='in')
    seed_state = xs.variable(global_name='seed_state', intent='out')

    def initialize(self):
        # instantiate a SeedSequence
        self.seed_seq = np.random.SeedSequence(entropy=(self.seed_entropy))
        self.seed_state = self.spawn_next(self.seed_seq)

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
