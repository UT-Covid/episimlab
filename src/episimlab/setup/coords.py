import xsimlab as xs
import xarray as xr


@xs.process
class InitDefaultCoords:
    """Example process that generates coordinates for each dimension
    in the counts array (except time).
    """

    age_group = xs.index(groups=['coords'], dims='age_group', global_name='age_group')
    risk_group = xs.index(groups=['coords'], dims='risk_group', global_name='risk_group')
    compartment = xs.index(groups=['coords'], dims='compartment', global_name='compartment')
    vertex = xs.index(groups=['coords'], dims='vertex', global_name='vertex')

    def initialize(self):
        self.age_group = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        self.vertex = range(3)
