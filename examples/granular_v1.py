import pandas as pd
import xsimlab as xs
import xarray as xr
from copy import deepcopy
from episimlab.models import PartitionFromTravel
import datetime
from episimlab.partition.travel_pat import TravelPatFromCSV
from episimlab.models import ExampleSIR, EpiModel
from episimlab.models.example_sir import SetupPhi
from episimlab.models.partition_v1 import *
from episimlab.compt_model import ComptModel
from episimlab.foi import BaseFOI
from episimlab.utils import get_var_dims, group_dict_by_var, visualize_compt_graph, coerce_to_da, fix_coord_dtypes, IntPerDay
from episimlab.setup.sto import SetupStochasticFromToggle
from episimlab.setup.seed import SeedGenerator
from episimlab.setup.state import SetupStateWithRiskFromCSV
import networkx as nx


@xs.process
class SetupStateFromCSV:
    """Initialize state matrix"""
    TAGS = ('granular', 'dependency::pandas')
    _coords = xs.group_dict('coords')
    state = xs.global_ref('state', intent='out')
    census_fp = xs.variable(intent='in', description='Path to CSV file from which to load initial susceptible population')

    def initialize(self):
        self.state = self.get_census_xa(df=self.get_census_df())
        self.state.loc[dict(compt='Pa', vertex='78712', age='20-49', risk='low')] = 1

    def get_census_df(self):
        return pd.read_csv(
            self.census_fp, dtype={'GEOID': str}
        ).drop('Unnamed: 0', axis=1).rename(columns={'GEOID': 'vertex', 'age_bin': 'age'})

    def get_census_xa(self, df: pd.DataFrame) -> xr.DataArray:

        df.set_index(['vertex', 'age', 'risk'], inplace=True)
        # filter to zcta that we want to model in the simulation (vertex coords)
        df = df.loc[self.coords['vertex']]
        da = xr.DataArray.from_series(df['estimate'])
        da.coords['age'] = da.coords['age'].astype(str)

        compt_fill = []
        df['compt'] = 'S'
        compt_fill.append(df)
        for c in ['E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D']:  # todo: pull from self.DIMS
            ccopy = deepcopy(df)
            ccopy['compt'] = c
            ccopy['estimate'] = 0.0  # remove the population
            compt_fill.append(ccopy)
        full_census = pd.concat(compt_fill)

        full_census = full_census.set_index([i for i in self.dims])
        census_xr = full_census.to_xarray()
        census_xr = census_xr.fillna(0.0)
        census_xa = census_xr['estimate']

        return census_xa

    @property
    def dims(self):
        return get_var_dims(ComptModel, 'state')

    @property
    def coords(self):
        return group_dict_by_var(self._coords)


class GranularFromTravel(EpiModel):
    """Nine-compartment SEIR model with partitioning from Episimlab V1"""
    TAGS = ('SEIR', 'compartments::9', 'contact-partitioning')
    DATA_DIR = './tests/data'
    PROCESSES = {
        # Core processes
        'compt_model': ComptModel,
        'setup_sto': SetupStochasticFromToggle,
        'setup_seed': SeedGenerator,
        'setup_compt_graph': SetupComptGraph,
        'int_per_day': IntPerDay,
        'setup_coords': SetupCoords,
        'setup_state': SetupStateWithRiskFromCSV,

        # Contact partitioning
        'setup_travel': TravelPatFromCSV,
        'setup_contacts': ContactsFromCSV,
        'partition': Partition,

        # Calculate greeks used by edge weight processes
        'setup_omega': SetupOmega,
        'setup_pi': SetupPiDefault,
        'setup_nu': SetupNuDefault,
        'setup_mu': mu.SetupStaticMuIh2D,
        'setup_gamma_Ih': gamma.SetupGammaIh,
        'setup_gamma_Ia': gamma.SetupGammaIa,
        'setup_gamma_Iy': gamma.SetupGammaIy,
        'setup_sigma': sigma.SetupStaticSigmaFromExposedPara,
        'setup_rho_Ia': rho.SetupRhoIa,
        'setup_rho_Iy': rho.SetupRhoIy,

        # Used for RateE2Pa and RateE2Py
        'rate_E2P': RateE2P,

        # All the expected edge weights
        'rate_S2E': RateS2E,
        'rate_E2Pa': RateE2Pa,
        'rate_E2Py': RateE2Py,
        'rate_Pa2Ia': RatePa2Ia,
        'rate_Py2Iy': RatePy2Iy,
        'rate_Ia2R': RateIa2R,
        'rate_Iy2R': RateIy2R,
        'rate_Iy2Ih': RateIy2Ih,
        'rate_Ih2R': RateIh2R,
        'rate_Ih2D': RateIh2D,
    }

    RUNNER_DEFAULTS = dict(
        clocks={
            'step': pd.date_range(start=datetime.datetime(2020, 2, 29), end=datetime.datetime(2020, 6, 30), freq='1D')
        },
        input_vars={
            'setup_sto__sto_toggle': 0,
            'setup_seed__seed_entropy': 12345,
            'rate_S2E__beta': 0.35,
            'rate_Iy2Ih__eta': 0.169492,
            'rate_E2Py__tau': 0.57,
            'rate_E2Pa__tau': 0.57,
            'setup_rho_Ia__tri_Pa2Ia': 2.3,
            'setup_rho_Iy__tri_Py2Iy': 2.3,
            'setup_sigma__tri_exposed_para': [1.9, 2.9, 3.9],
            'setup_gamma_Ih__tri_Ih2R': [9.4, 10.7, 12.8],
            'setup_gamma_Ia__tri_Iy2R_para': [3.0, 4.0, 5.0],
            'setup_mu__tri_Ih2D': [5.2, 8.1, 10.1],
            'travel_pat_fp': 'data/granular_data/2020_travel_age_risk_for_contact_partitioning_agm_2022_demo_first_wave.csv',
            'contacts_fp': 'data/granular_data/polymod_dummy_ages.csv',
            'census_fp': 'data/granular_data/atx_zcta_high_risk_long.csv'
        },
        output_vars={
            'compt_model__state': 'step'
        }
    )

    def plot(self, show=True):
        plot = self.out_ds['compt_model__state'].sum(['age', 'risk', 'vertex']).plot.line(x='step', aspect=2, size=9)
        if show:
            plt.show()


def main():

    austin_model = GranularFromTravel()
    output = austin_model.run()

    return output


if __name__ == '__main__':

    output = main()
    output.to_zarr(store='test_granular_agm_3.zarr')
