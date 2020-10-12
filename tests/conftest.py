import pytest
import numpy as np
import logging
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


@pytest.fixture()
def phi_midx_mapping(counts_dims, counts_coords) -> xr.DataArray:
    """A MultiIndex mapping for a phi array"""
    dims = ['age_group', 'risk_group']
    coords = {
        k: v for k, v in counts_coords.items()
        if k in dims
    }
    shape = np.array([len(coords[d]) for d in dims])
    data = np.arange(np.prod(shape)).reshape(shape)
    # logging.debug(f"data: {data}")
    # logging.debug(f"dims: {dims}")
    # logging.debug(f"coords: {coords}")
    return xr.DataArray(data=data, dims=dims, coords=coords)

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

