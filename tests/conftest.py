import pytest
import xarray as xr


@pytest.fixture()
def counts_dims():
    return ['vertex', 'age_group', 'risk_group', 'compartment']

@pytest.fixture()
def counts_coords():
    return {
        'vertex': range(3),
        'age_group': ['0-4', '5-17', '18-49', '50-64', '65+'],
        'risk_group': ['low', 'high'],
        'compartment': ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                        'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                        'Py2Iy', 'Iy2Ih', 'H2D']
    }

@pytest.fixture()
def counts_basic(counts_dims, counts_coords):
    """
    """
    return xr.DataArray(
        data=1.,
        dims=counts_dims,
        coords=counts_coords,
    )


