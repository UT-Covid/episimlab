import pandas as pd
import xsimlab as xs
from matplotlib.pyplot import plt


class EpiModel(xs.Model):
    """Lightweight subclass of `xsimlab.Model`. Provides a framework for
    testing, running, and plotting an epidemiological model with default
    arguments
    """
    PROCESSES = {}
    RUNNER_DEFAULTS = {
        'clocks': {
            'step': pd.date_range(start='3/1/2020', end='3/2/2020', freq='24H')
        },
        'input_vars': dict(),
        'output_vars': dict()
    }

    def __init__(self, processes: dict = None):
        if processes is None:
            processes = EpiModel.PROCESSES.copy()
        super(EpiModel, self).__init__(processes)
        assert not hasattr(self, 'processes')
        assert not hasattr(self, 'input_vars')
        assert not hasattr(self, 'output_vars')
        assert not hasattr(self, 'clocks')
        assert not hasattr(self, 'in_ds')
        assert not hasattr(self, 'out_ds')
    
    def run_with_defaults(self, **kwargs) -> xr.Dataset:
        self.in_ds = self.default_in_ds(**kwargs)
        self.out_ds = self.in_ds.xsimlab.run(model=self)
        return self.out_ds

    @property
    def default_in_ds(self, **kwargs) -> xs.Dataset:
        setup_kw = self.RUNNER_DEFAULTS.copy()
        setup_kw.update(kwargs)
        return xs.create_setup(model=self, **setup_kw)
