import pandas as pd
import xsimlab as xs


@xs.process
class SetupToyCoords:
    """Initialize state coordinates"""
    TAGS = ('coords', 'model::SIRV', 'example')
    compt = xs.variable(global_name='compt_coords', groups=['coords'], intent='out')
    age = xs.variable(global_name='age_coords', groups=['coords'], intent='out')
    risk = xs.variable(global_name='risk_coords', groups=['coords'], intent='out')
    vertex = xs.variable(global_name='vertex_coords', groups=['coords'], intent='out')
    
    def initialize(self):
        self.compt = ['S', 'I', 'R', 'V'] 
        self.age = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk = ['low', 'high']
        self.vertex = ['Austin', 'Houston', 'San Marcos', 'Dallas']


@xs.process
class SetupCoordsFromTravel(SetupToyCoords):
    TAGS = ('deprecated', 'coords')
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
        age = pd.unique(df[['age', 'age_dest']].values.ravel('K'))
        vertex = pd.unique(df[['source', 'destination']].values.ravel('K'))
        return dict(
            age=age,
            vertex=vertex
        )

    def initialize(self):
        raise DeprecationWarning()
        for dim, coord in self.get_df_coords().items():
            setattr(self, dim, coord)
        self.risk = ['low', 'high']
        self.compartment = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                            'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                            'Py2Iy', 'Iy2Ih', 'H2D']
        
