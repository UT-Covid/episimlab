import pandas as pd
import xsimlab as xs


@xs.process
class SetupToyCoords:
    """Initialize state coordinates. Imports compartment coordinates from the
    compartment graph.
    """
    TAGS = ('coords', 'example')

    compt = xs.index(dims=('compt'), global_name='compt_coords', groups=['coords'], 
                     description='Coordinates for the `compt` dimension. In '
                     'other words, the list of compartment names in the model.')
    age = xs.index(dims=('age'), global_name='age_coords', groups=['coords'],
                   description='Coordinates for the `age` dimension.')
    risk = xs.index(dims=('risk'), global_name='risk_coords', groups=['coords'],
                    description='Coordinates for the `risk` dimension.')
    vertex = xs.index(dims=('vertex'), global_name='vertex_coords', groups=['coords'],
                      description='Coordinates for the `vertex` dimension.')
    compt_graph = xs.global_ref('compt_graph', intent='in')
    
    def initialize(self):
        self.compt = self.compt_graph.nodes
        self.age = ['0-4', '5-17', '18-49', '50-64', '65+']
        self.risk = ['low', 'high']
        self.vertex = ['Austin', 'Houston', 'San Marcos', 'Dallas']