import pytest
import numpy as np
import pandas as pd
import logging
import xarray as xr


@pytest.fixture()
def counts_dims():
    return ['vertex', 'age_group', 'risk_group', 'compartment']


@pytest.fixture(params=[
    3,
    # 5
])
def counts_coords(request):
    return {
        'vertex': range(request.param),
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
    # 'ones'
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


@pytest.fixture()
def adj_grp_mapping(counts_dims, counts_coords) -> xr.DataArray:
    """A MultiIndex mapping for an adj array"""
    dims = ('vertex', 'age_group', 'risk_group', 'compartment')
    coords = {
        k: v for k, v in counts_coords.items()
        if k in dims
    }
    shape = np.array([len(coords[d]) for d in dims])
    data = np.arange(np.prod(shape)).reshape(shape)
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def phi_grp_mapping(counts_dims, counts_coords) -> xr.DataArray:
    """A MultiIndex mapping for a phi array"""
    dims = ['age_group', 'risk_group']
    coords = {
        k: v for k, v in counts_coords.items()
        if k in dims
    }
    shape = np.array([len(coords[d]) for d in dims])
    data = np.arange(np.prod(shape)).reshape(shape)
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def adj_t(counts_coords):
    dims = ('adj_grp1', 'adj_grp2')
    # we assume that all the dimensions in counts are also dimensions
    # on the adjacency matrix
    idx_size = np.product([
        len(coord) for coord in counts_coords.values()
    ])
    return xr.DataArray(
        data=0.1,
        dims=dims,
        coords={dim: range(idx_size) for dim in dims}
    )


@pytest.fixture()
def phi_t():
    data = [[0.51540028, 0.51540028, 0.94551748, 0.94551748, 1.96052056, 1.96052056, 0.12479711, 0.12479711, 0.0205698, 0.0205698 ],
            [0.51540028, 0.51540028, 0.94551748, 0.94551748, 1.96052056, 1.96052056, 0.12479711, 0.12479711, 0.0205698, 0.0205698 ],
            [0.20813759, 0.20813759, 1.72090425, 1.72090425, 1.9304265, 1.9304265, 0.16597259, 0.16597259, 0.0238168, 0.0238168 ],
            [0.20813759, 0.20813759, 1.72090425, 1.72090425, 1.9304265, 1.9304265, 0.16597259, 0.16597259, 0.0238168, 0.0238168 ],
            [0.24085226, 0.24085226, 0.90756038, 0.90756038, 1.68238057, 1.68238057, 0.23138952, 0.23138952, 0.0278581, 0.0278581 ],
            [0.24085226, 0.24085226, 0.90756038, 0.90756038, 1.68238057, 1.68238057, 0.23138952, 0.23138952, 0.0278581, 0.0278581 ],
            [0.20985118, 0.20985118, 0.70358752, 0.70358752, 1.24247158, 1.24247158, 0.97500204, 0.97500204, 0.10835478, 0.10835478 ],
            [0.20985118, 0.20985118, 0.70358752, 0.70358752, 1.24247158, 1.24247158, 0.97500204, 0.97500204, 0.10835478, 0.10835478 ],
            [0.14845117, 0.14845117, 0.69386045, 0.69386045, 0.98826341, 0.98826341, 0.34871121, 0.34871121, 0.61024946, 0.61024946 ],
            [0.14845117, 0.14845117, 0.69386045, 0.69386045, 0.98826341, 0.98826341, 0.34871121, 0.34871121, 0.61024946, 0.61024946 ]]
    dims = ['phi_grp1', 'phi_grp2']
    coords = {
        'phi_grp1': range(10),
        'phi_grp2': range(10)
    }
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def epis(rho, gamma, sigma, pi, eta, nu, mu, tau):
    return dict(
        rho=rho, gamma=gamma, sigma=sigma, pi=pi,
        eta=eta, nu=nu, mu=mu, tau=tau
    )


@pytest.fixture()
def rho(counts_coords):
    data = 0.43478261
    dims = ('age_group', 'compartment')
    coords = {k: counts_coords[k] for k in dims}
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def gamma(counts_coords):
    data = 0.
    dims = ['compartment']
    coords = {k: counts_coords[k] for k in dims}
    da = xr.DataArray(data=data, dims=dims, coords=coords)
    da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
    return da


@pytest.fixture()
def sigma():
    return 0.34482759


@pytest.fixture()
def pi(counts_coords):
    # pi = np.array(
        # [[5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
         # [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]])
    data = np.array([
        [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
        [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]
    ])
    dims = ('risk_group', 'age_group')
    coords = {k: counts_coords[k] for k in dims}
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def eta():
    return 0.169492


@pytest.fixture()
def nu(counts_coords):
    dims = ['age_group']
    return xr.DataArray(
        [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128],
        dims=dims,
        coords={k: counts_coords[k] for k in dims}
    )


@pytest.fixture()
def mu():
    return 0.12820513


@pytest.fixture()
def beta():
    return 0.035


@pytest.fixture()
def tau():
    return 0.57


@pytest.fixture(params=[
    -1,
    # TODO
    0,
    # 5,
    # 20
])
def sto_toggle(request):
    return request.param


@pytest.fixture()
def omega(counts_coords):
    # omega_a = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667])
    # omega_y = np.array([1.        , 1.        , 1.        , 1.        , 1.        ])
    # omega_h = np.array([0.        , 0.        , 0.        , 0.        , 0.        ])
    # omega_pa = np.array([0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149])
    # omega_py = np.array([1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724])
    data = np.array([[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
                     [1.        , 1.        , 1.        , 1.        , 1.        ],
                     [0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149],
                     [1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724]])
    dims = ['age_group', 'compartment']
    coords = {k: counts_coords[k] for k in dims}

    da = xr.DataArray(data=0., dims=dims, coords=coords)
    da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
    assert isinstance(da, xr.DataArray), type(da)
    return da


@pytest.fixture()
def seed_entropy():
    return 12345


@pytest.fixture(params=[
    True,
    False
])
def stochastic(request):
    return request.param


@pytest.fixture(params=[
    # int_per_day == 1
    # TODO
    # '24H',
    # int_per_day == 2
    '12H'
])
def step_clock(request):
    return {
        'step': pd.date_range(
            start='1/1/2018', end='1/15/2018', freq=request.param
        )
    }


@pytest.fixture(params=[
    # int_per_day == 1
    # TODO
    # 24,
    # int_per_day == 2
    12
])
def step_delta(request):
    try:
        return np.timedelta64(request.param, 'h')
    except ValueError:
        logging.debug(f"type(request.param): {type(request.param)}")
        raise
