import xsimlab as xs
import xarray as xr
import numpy as np
import yaml

from ..setup import seed, epi, sto


@xs.process
class ReadV1Config:
    """Reads model input variables from a version 1.0 configuration file.
    By default, this is a .yaml file at a filepath defined by input variable
    `config_fp`.
    """
    config_fp = xs.variable(static=True, intent='in')

    seed_entropy = xs.foreign(seed.SeedGenerator, 'seed_entropy', intent='out')
    sto_toggle = xs.foreign(sto.InitStochasticFromToggle,
                            'sto_toggle', intent='out')
    t_onset_to_h = xs.foreign(
        epi.SetupEtaFromAsympRate, 't_onset_to_h', intent='out')
    tri_h2r = xs.foreign(epi.SetupStaticGamma, 'tri_h2r', intent='out')
    tri_h2d = xs.foreign(epi.SetupStaticMuFromHtoD, 'tri_h2d', intent='out')
    tri_y2r_para = xs.foreign(epi.SetupStaticGamma,
                              'tri_y2r_para', intent='out')
    tri_py2iy = xs.foreign(epi.SetupStaticRhoFromTri, 'tri_py2iy', intent='out')
    tri_exposed_para = xs.foreign(epi.SetupStaticSigmaFromExposedPara,
                                  'tri_exposed_para', intent='out')
    tri_pa2ia = xs.foreign(epi.SetupStaticRhoFromTri, 'tri_pa2ia', intent='out')
    asymp_relative_infect = xs.foreign(
        epi.SetupStaticOmega, 'asymp_relative_infect', intent='out')
    asymp_rate = xs.foreign(epi.SetupTauFromAsympRate,
                            'asymp_rate', intent='out')

    def initialize(self):
        config = self.get_config()
        for var in config:
            setattr(self, var, config[var])

    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
