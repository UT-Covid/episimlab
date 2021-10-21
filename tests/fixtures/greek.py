import pytest
import numpy as np
import xarray as xr


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
def rho(counts_coords):
    data = 0.43478261
    dims = ['compartment']
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



@pytest.fixture()
def symp_h_ratio_w_risk(counts_coords):
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [[0.0002791, 0.0002146, 0.0132154, 0.0285634, 0.0338733],
            [0.002791, 0.002146, 0.132154, 0.285634, 0.338733]]


@pytest.fixture()
def symp_h_ratio(counts_coords):
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [0.00070175, 0.00070175, 0.04735258, 0.16329827, 0.25541833]


@pytest.fixture()
def prop_trans_in_p():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.44


@pytest.fixture()
def hosp_f_ratio(counts_coords) -> xr.DataArray:
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [0.04, 0.12365475, 0.03122403, 0.10744644, 0.23157691]


@pytest.fixture()
def asymp_relative_infect():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.666666666


@pytest.fixture()
def tri_h2d():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [5.2, 8.1, 10.1]


@pytest.fixture()
def tri_h2r():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_exposed_para():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [1.9, 2.9, 3.9]


@pytest.fixture()
def tri_pa2ia():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 2.3


@pytest.fixture()
def tri_py2iy():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 2.3


@pytest.fixture()
def asymp_rate():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.43


@pytest.fixture()
def t_onset_to_h():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 5.9


@pytest.fixture()
def tri_h2r():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_y2r_para():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [3.0, 4.0, 5.0]