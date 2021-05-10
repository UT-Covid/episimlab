import yaml
import xsimlab as xs
import xarray as xr


@xs.process
class InitDefaultCoords:
    """Example process that generates coordinates for each dimension
    in the counts array (except time).
    """

    age_group = xs.index(dims='age_group', global_name='age_group')
    risk_group = xs.index(dims='risk_group', global_name='risk_group')
    compartment = xs.index(dims='compartment', global_name='compartment')
    vertex = xs.index(dims='vertex', global_name='vertex')

    def initialize(self):
        self.age_group = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        self.vertex = range(3)


@xs.process
class InitCoordsExpectVertex(InitDefaultCoords):
    """InitDefaultCoords but allows user to pass `vertex_labels`."""
    vertex_labels = xs.variable(dims='vertex', intent='in')

    def initialize(self):
        self.age_group = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        self.vertex = self.vertex_labels


@xs.process
class InitCoordsFromConfig(InitDefaultCoords):
    """InitDefaultCoords but allows user to pass all coordinates in YAML
    file at `config_fp`.
    """
    config_fp = xs.variable(static=True, intent='in')

    def initialize(self):
        cfg = self.get_config()
    
    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.safe_load(f)