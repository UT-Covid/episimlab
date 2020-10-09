import pytest
import xarray as xr
from episimlab.seir import BruteForceFOI


class TestFOIBruteForce:

    def test_can_import(self, counts_basic):
        """
        """
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            'beta': 0.035,
            'omega': xr.DataArray(
                data=[0.667, 1., 0.9, 1.3],
                # TODO: age dim
                dims='compartment',
                coords=dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])
            ),
            'counts': counts_basic,
            'phi_t': [
                [0.51540028, 0.94551748, 1.96052056, 0.12479711, 0.0205698 ],
                [0.20813759, 1.72090425, 1.9304265 , 0.16597259, 0.0238168 ],
                [0.24085226, 0.90756038, 1.68238057, 0.23138952, 0.0278581 ],
                [0.20985118, 0.70358752, 1.24247158, 0.97500204, 0.10835478],
                [0.14845117, 0.69386045, 0.98826341, 0.34871121, 0.61024946]]

        }
        foi_getter = BruteForceFOI(**inputs)
        foi_getter.run_step()
