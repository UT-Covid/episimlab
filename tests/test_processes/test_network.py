import pytest
from episimlab.processes import network


class TestNetwork:

    def test_can_initialize(self):
        """
        """
        ntwk = network.Network(amx=range(10))
        ntwk.initialize()
        assert ntwk.amx

