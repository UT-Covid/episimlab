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
    
    def input_vars_from_config(self, config_fp: str) -> dict:
        if config_fp is None:
            if hasattr(self, 'config_fp'):
                config_fp = self.config_fp
            else:
                logging.info('No path to config (`config_fp`) was specified. Using model defaults.')
                return dict()
        with open(config_fp, 'r') as f:
            return yaml.safe_load(f)
    
    def parse_input_vars(self, d: dict) -> dict:
        """Parse dictionary of input variables, returning a modified dictionary.
        Attempts to parse keys without the xsimlab-canonical double underscore
        denoting {process}__{variable}, dynamically assigning variables to 
        processes that ingest them.
        """
        mod = dict()
        for k, v in d.items():
            used = False
            if '__' in k:
                mod[k] = v
            else:
                for proc, name in self.input_vars:
                    if name == k:
                        used = True
                        mod[f"{proc}__{name}"] = v
                if not used:
                    raise ValueError(
                        f"Could not find a process that ingests variable named "
                        f"{k}. Expected input variables are {self.input_vars}.")
        return mod
            
    def run(self, **kwargs) -> xr.Dataset:
        self.in_ds = self.get_in_ds(**kwargs)
        self.out_ds = self.in_ds.xsimlab.run(model=self)
        return self.out_ds

    def get_in_ds(self, config_fp: str = None, input_vars: dict = None, **kwargs) -> xr.Dataset:
        setup_kw = getattr(self, 'RUNNER_DEFAULTS', dict()).copy()
        setup_kw.update(kwargs)
        setup_kw['input_vars'].update(self.parse_input_vars(
            self.input_vars_from_config(config_fp=config_fp)))
        if input_vars is not None:
            setup_kw['input_vars'].update(self.parse_input_vars(input_vars))
        setup_kw['input_vars'] = self.parse_input_vars(setup_kw['input_vars'])
        return xs.create_setup(model=self, **setup_kw)
