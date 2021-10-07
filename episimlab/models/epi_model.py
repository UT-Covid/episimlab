import logging
import yaml
import pandas as pd
import xarray as xr
import xsimlab as xs
from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO)


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
            processes = self.PROCESSES.copy()
        super(EpiModel, self).__init__(processes)
        assert not hasattr(self, 'in_ds')
        assert not hasattr(self, 'out_ds')
    
    def input_vars_from_config(self, config_fp=None) -> dict:
        if config_fp is None:
            if hasattr(self, 'config_fp'):
                config_fp = self.config_fp
            else:
                logging.info('No path to config (`config_fp`)was specified. Using model defaults.')
                return dict()
        with open(config_fp, 'r') as f:
            return yaml.safe_load(f)
            
    def run(self, config_fp=None, input_vars=None, **kwargs) -> xr.Dataset:
        try:
            kwargs['input_vars'] = self.input_vars_from_config(config_fp=config_fp)
        except:
            raise
        if input_vars is not None:
            kwargs['input_vars'].update(input_vars)
        if not kwargs['input_vars']:
            del kwargs['input_vars']

        self.in_ds = self.get_in_ds(**kwargs)
        self.out_ds = self.in_ds.xsimlab.run(model=self)
        return self.out_ds

    def get_in_ds(self, **kwargs) -> xr.Dataset:
        setup_kw = self.RUNNER_DEFAULTS.copy()
        setup_kw.update(kwargs)
        return xs.create_setup(model=self, **setup_kw)
