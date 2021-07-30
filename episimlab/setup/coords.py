import yaml
import numpy as np
import pandas as pd
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
class InitVaccineCoordsVertex(InitDefaultCoords):
    """Adds vaccine-related compartments with user specified-vertices"""

    vertex = xs.index(dims='vertex', global_name='vertex')

    def initialize(self):
        self.age_group = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D', 'V', 'EV', 'V2Ev', 'Ev2P']
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
        coords = self.get_config()['coords']
        assert isinstance(coords, dict)
        for dim, coord in coords.items():
            setattr(self, dim, coord)
    
    def get_config(self) -> dict:
        with open(self.config_fp, 'r') as f:
            return yaml.safe_load(f)


@xs.process
class InitCoordsExceptVertex:
    age_group = xs.index(dims='age_group', global_name='age_group')
    risk_group = xs.index(dims='risk_group', global_name='risk_group')
    compartment = xs.index(dims='compartment', global_name='compartment')

    def initialize(self):
        self.age_group = ['<5', '5-17', '18-49', '50-64', '65+']
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']


@xs.process
class InitCoordsFromTravel(InitDefaultCoords):
    travel_fp = xs.variable(intent='in')

    def load_travel_df(self):
        tdf = pd.read_csv(self.travel_fp)
        tdf['date'] = pd.to_datetime(tdf['date'])
        try:
            tdf = tdf.rename(columns={'age_src': 'age'})
        except KeyError:
            raise
            # pass
        return tdf

    def get_df_coords(self) -> dict:
        df = self.load_travel_df()
        age_group = pd.unique(df[['age', 'age_dest']].values.ravel('K'))
        vertex = pd.unique(df[['source', 'destination']].values.ravel('K'))
        # breakpoint()
        return dict(
            age_group=age_group,
            vertex=vertex
        )

    def initialize(self):
        for dim, coord in self.get_df_coords().items():
            setattr(self, dim, coord)
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']

@xs.process
class InitCoordsFromTravelVaccine(InitCoordsFromTravel):
    """
    Extending to initialize some additional compartments
    """

    def initialize(self):
        for dim, coord in self.get_df_coords().items():
            setattr(self, dim, coord)
        self.risk_group = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'V', 'V2Ev', 'Ev2Pa', 'Ev2Py',
                            'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2R', 'H2D']
