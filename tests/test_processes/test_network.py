import pytest
from episimlab.processes import network


class TestNetwork:

    def test_can_initialize(self, cur_cts):
        """
        """
        ntwk = network.Network(amx=range(10), cur_cts=cur_cts)
        ntwk.run_step()
        assert ntwk.amx

