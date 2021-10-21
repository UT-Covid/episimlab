import os
import yaml
import pytest
import numpy as np
import pandas as pd
import logging
import xarray as xr

from .fixtures.greek import *

@pytest.fixture(params=[
    3,
    # 5
])
def coords(request):
    return {
        ('proc_name', 'vertex'): list(range(request.param)),
        ('proc_name', 'age'): ['0-4', '5-17', '18-49', '50-64', '65+'],
        ('proc_name', 'risk'): ['low', 'high'],
        ('proc_name', 'compt'): ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                        'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                        'Py2Iy', 'Iy2Ih', 'H2D']
    }

@pytest.fixture(params=[
    3,
    5
])
def counts_coords(request):
    return {
        'vertex': list(range(request.param)),
        'age_group': ['0-4', '5-17', '18-49', '50-64', '65+'],
        'risk_group': ['low', 'high'],
        'compartment': ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                        'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                        'Py2Iy', 'Iy2Ih', 'H2D']
    }


@pytest.fixture(params=[
    0.75,
])
def foi(request, counts_coords):
    dims = ('vertex', 'age_group', 'risk_group')
    return xr.DataArray(
        data=request.param,
        dims=dims,
        coords={dim: counts_coords[dim] for dim in dims}
    )


@pytest.fixture(params=[
    'realistic',
    # 'mid_sim',
    'ones'
])
def counts_basic(counts_dims, counts_coords, request):
    if request.param == 'realistic':
        da = xr.DataArray(
            data=0.,
            dims=counts_dims,
            coords=counts_coords,
        )
        # Houston
        da.loc[dict(vertex=0, compartment='S')] = 2.326e6 / 10.
        # Austin
        da.loc[dict(vertex=1, compartment='S')] = 1e6 / 10.
        # Beaumont
        da.loc[dict(vertex=2, compartment='S')] = 1.18e5 / 10.
        # Start with 50 infected asymp in Austin
        da.loc[dict(vertex=1, compartment='Ia')] = 50.
    elif request.param == 'ones':
        da = xr.DataArray(
            data=1.,
            dims=counts_dims,
            coords=counts_coords,
        )
    elif request.param == 'mid_sim':
        raise NotImplementedError()
    else:
        raise ValueError()
    return da

@pytest.fixture(params=[
    # deterministic at all steps
    -1,
    # stochastic at all steps
    0,
    5,
    20
])
def sto_toggle(request):
    return request.param

@pytest.fixture()
def seed_entropy():
    return 12345


@pytest.fixture()
def seed_state(seed_entropy):
    return seed_entropy


@pytest.fixture(params=[
    True,
    False
])
def stochastic(request):
    return request.param


@pytest.fixture(params=[
    # int_per_day == 1
    '24H'
])
def step_clock(request):
    return {
        'step': pd.date_range(
            start='1/1/2018', end='1/15/2018', freq=request.param
        )
    }


@pytest.fixture(params=[
    24
])
def step_delta(request):
    try:
        return np.timedelta64(request.param, 'h')
    except ValueError:
        logging.debug(f"type(request.param): {type(request.param)}")
        raise


@pytest.fixture
def config_fp(tmpdir):
    """Fixture factory that writes a dictionary to a YAML file and
    provides the file path.
    """
    fp = tmpdir.mkdir("config").join("config.yaml")

    def _config_fp(config_dict):
        assert not os.path.isfile(fp)
        with open(fp, 'w') as f:
            yaml.dump(config_dict, f)
        assert os.path.isfile(fp)
        return str(fp)

    return _config_fp
