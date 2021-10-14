import xsimlab
# Import an existing SIR model 
from episimlab.models import ExampleSIR
from episimlab.utils import visualize_compt_graph
import networkx as nx



@xsimlab.process
class CustomRecoveryRate:
    """A single process in the model. Calculates a recovery rate (`rate_I2R`)."""
    # Variables output by this process (intent='out')
    rate_I2R = xsimlab.variable(
        global_name='rate_I2R', groups=['edge_weight'], intent='out', 
        description="rate of change from compartments I to R")

    # Variables ingested by this process (intent='in')
    gamma = xsimlab.variable(global_name='gamma', intent='in')

    # The state is the N-D array of every individual in the simulation,
    # stratified by age, risk group, geospatial identifier, compartment, etc.
    state = xsimlab.global_ref('state', intent='in')

    def run_step(self):
        """Re-calculate `rate_I2R` at every step of the simulation. We can
        instead decide to calculate only once at the beginning by renaming
        this method `initialize`. We calculate recovery rate as the product
        of `gamma` and the current state of the `I` compartment.
        """
        self.rate_I2R = self.gamma * self.state.loc[{'compt': 'I'}]


@xsimlab.process
class CustomSetupComptGraph:
    """A single process in the model. Defines the directed graph `compt_graph`
    that defines the compartments and allowed transitions between them.
    """
    compt_graph = xsimlab.global_ref('compt_graph', intent='out')

    def initialize(self):
        """This method is run once at the beginning of the simulation."""
        self.compt_graph = self.get_compt_graph()

    def finalize(self):
        """This method is run once at the end of the simulation."""
        self.visualize(path='./examples/example_sir_compt_graph.svg')

    def get_compt_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from([
            ('S', {"color": "red"}),
            ('I', {"color": "blue"}),
            ('R', {"color": "green"}),
        ])
        g.add_edges_from([
            ('S', 'I', {"priority": 0, "color": "red"}),
            ('I', 'R', {"priority": 1, "color": "blue"}),
        ])
        return g
    
    def visualize(self, path=None):
        """Visualize the compartment graph, saving as a file at `path`"""
        return visualize_compt_graph(self.compt_graph, path=path)


# Instantiate the included model
model = ExampleSIR()

# Visualize the processes in the model (requires graphviz)
model.visualize()

# Edit the model so it uses our custom process
model = model.update_processes({
    'rate_I2R': CustomRecoveryRate, 
    'setup_compt_graph': CustomSetupComptGraph})

# We can optionally overwrite model defaults from a YAML configuration file
model.config_fp = './examples/example_sir.yaml'

# Run the model using Dask and xarray-simlab (xsimlab)
results = model.run(input_vars={
    # We can optionally overwrite specific input variables at runtime
    'beta': 0.085
})

# Plot the state over time
model.plot()

# The final state is an N-D array represented in Python as an xarray.DataArray
# object (like a numpy array, but with labeled axes)
final_state = results['compt_model__state']
print(final_state)
# <xarray.DataArray 'compt_model__state' (step: 15, vertex: 4, compt: 4, age: 5, risk: 2)>
# array([[[[[200., 200.],
#           [200., 200.],
#           ...
#           [  0.,   0.],
#           [  0.,   0.]]]]])
# Coordinates:
#   * age      (age) object '0-4' '5-17' '18-49' '50-64' '65+'
#   * compt    (compt) object 'S' 'I' 'R' 'V'
#   * risk     (risk) object 'low' 'high'
#   * step     (step) datetime64[ns] 2020-03-01 2020-03-02 ... 2020-03-15
#   * vertex   (vertex) object 'Austin' 'Houston' 'San Marcos' 'Dallas'