import xsimlab as xs
import xarray as xr
import numpy as np
import logging
import yaml
from collections.abc import Iterable

from ..setup import seed, epi, sto, coords
from ..utils import get_var_dims


@xs.process
class ReadV1Config:
    """Reads model input variables from a version 1.0 configuration file.
    By default, this is a .yaml file at a filepath defined by input variable
    `config_fp`.
    """
    KEYS_MAPPING = {
        'symp_h_ratio_w_risk': epi.SetupStaticPi,
        'seed_entropy': seed.SeedGenerator,
        't_onset_to_h': epi.SetupEtaFromAsympRate,
        'tri_h2r': epi.SetupStaticGamma,
        'sto_toggle': sto.InitStochasticFromToggle,
        'tri_h2d': epi.SetupStaticMuFromHtoD,
        'tri_y2r_para': epi.SetupStaticGamma,
        'tri_py2iy': epi.SetupStaticRhoFromTri,
        'tri_exposed_para': epi.SetupStaticSigmaFromExposedPara,
        'tri_pa2ia': epi.SetupStaticRhoFromTri,
        'asymp_relative_infect': epi.SetupStaticOmega,
        'asymp_rate': epi.SetupTauFromAsympRate,
        'hosp_f_ratio': epi.SetupStaticNu,
        'prop_trans_in_p': epi.SetupStaticOmega,
        'symp_h_ratio': epi.SetupStaticOmega,
    }

    config_fp = xs.variable(static=True, intent='in')
    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    # vertex = xs.global_ref('vertex')

    seed_entropy = xs.foreign(KEYS_MAPPING['seed_entropy'], 'seed_entropy',
                              intent='out')
    sto_toggle = xs.foreign(KEYS_MAPPING['sto_toggle'], 'sto_toggle',
                            intent='out')
    t_onset_to_h = xs.foreign(KEYS_MAPPING['t_onset_to_h'], 't_onset_to_h',
                              intent='out')
    tri_h2r = xs.foreign(KEYS_MAPPING['tri_h2r'], 'tri_h2r', intent='out')
    tri_h2d = xs.foreign(KEYS_MAPPING['tri_h2d'], 'tri_h2d', intent='out')
    tri_y2r_para = xs.foreign(KEYS_MAPPING['tri_y2r_para'], 'tri_y2r_para',
                              intent='out')
    tri_py2iy = xs.foreign(KEYS_MAPPING['tri_py2iy'], 'tri_py2iy', intent='out')
    tri_exposed_para = xs.foreign(KEYS_MAPPING['tri_exposed_para'],
                                  'tri_exposed_para', intent='out')
    tri_pa2ia = xs.foreign(KEYS_MAPPING['tri_pa2ia'], 'tri_pa2ia', intent='out')
    asymp_relative_infect = xs.foreign(KEYS_MAPPING['asymp_relative_infect'],
                                       'asymp_relative_infect', intent='out')
    asymp_rate = xs.foreign(KEYS_MAPPING['asymp_rate'], 'asymp_rate',
                            intent='out')
    hosp_f_ratio = xs.foreign(KEYS_MAPPING['hosp_f_ratio'], 'hosp_f_ratio',
                              intent='out')
    prop_trans_in_p = xs.foreign(KEYS_MAPPING['prop_trans_in_p'],
                                 'prop_trans_in_p', intent='out')
    symp_h_ratio = xs.foreign(KEYS_MAPPING['symp_h_ratio'], 'symp_h_ratio',
                              intent='out')
    symp_h_ratio_w_risk = xs.foreign(KEYS_MAPPING['symp_h_ratio_w_risk'],
                                     'symp_h_ratio_w_risk', intent='out')

    def initialize(self):
        config = self.get_config()
        for name, value in config.items():
            setattr(self, name, self.try_coerce_to_da(value=value, name=name))

    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def try_coerce_to_da(self, name, value):
        """Given a variable with `name`, and `value` set from a config file,
        retrieve the variable metadata and use it to coerce the `value` into
        an `xarray.DataArray` with the correct dimensions and coordinates.
        Returns `value` if variable is scalar (zero length dims attribute),
        DataArray otherwise.
        """
        # get dims
        dims = get_var_dims(self.KEYS_MAPPING[name], name)
        if not dims:
            return value
        # get coords
        coords = {dim: getattr(self, dim) for dim in dims if dim != 'value'}
        return xr.DataArray(data=value, dims=dims, coords=coords)
