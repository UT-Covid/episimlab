import xsimlab as xs
import logging


@xs.process
class SetupStochasticFromToggle:
    """Switches on stochasticity after simulation has run `sto_toggle` steps.
    """
    sto_toggle = xs.variable(static=True, global_name='sto_toggle', intent='in')
    stochastic = xs.global_ref('stochastic', intent='out')

    def initialize(self):
        """Ensures that stochastic is set during initialization"""
        self.run_step(step=0)

    @xs.runtime(args="step")
    def run_step(self, step):
        if self.sto_toggle == -1:
            self.stochastic = False
        elif step >= self.sto_toggle:
            self.stochastic = True
        else:
            self.stochastic = False
