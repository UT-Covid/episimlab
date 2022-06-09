import logging
import pandas as pd
import xarray as xr
import xsimlab as xs
from ..utils import group_dict_by_var, get_int_per_day, fix_coord_dtypes, get_var_dims
from .travel_pat import TravelPatFromCSV
from episimlab.setup.state import SetupStateWithRiskFromCSV

@xs.process
class BehaviorChange(TravelPatFromCSV):

    census_fp = xs.global_ref('census_fp', intent='in')
    state = xs.global_ref('state', intent='in')
    hosp_catchment_fp = xs.variable(
        static=True, intent='in', description="path to a CSV file containing hospital catchments"
    )

    def run_step(self, step_start, step_end):
        travel_df = self.get_travel_df()

        # Both step_start and step_end will be None for initialize
        if step_start is None and step_end is None:
            travel_df = travel_df[travel_df['date'] == travel_df['date'].min()]
        else:
            travel_df = travel_df[self.get_date_mask(travel_df['date'], step_start, step_end)]

        # Validation
        if travel_df.empty:
            raise ValueError(f'No travel data between {step_start} and {step_end}')
        logging.info(f'The date in Partition.get_travel_df is {travel_df["date"].unique()}')

        hosp_catchment = self.get_hospital_catchment()
        symp_catchment = self.get_symp_catchment()
        kappa = self.get_granular_kappa()

        updated_df = self.change_travel_inf(travel_df, hosp_catchment, symp_catchment, kappa)

        self.travel_pat = self.get_travel_da(updated_df)

    def get_hospital_catchment(self):
        """ load a CSV of hospital catchments
        """
        hosp_catch = pd.read_csv(self.hosp_catchment_fp)

        return hosp_catch

    def get_symp_catchment(self):

        # return a dataframe where vertex_src == vertex_dest
        unique_vertex = self.travel_pat.coords['vertex0'].values
        symp_catch = pd.DataFrame(
            {'vertex_src': unique_vertex,
             'vertex_dest': unique_vertex}
        )

        return symp_catch

    def get_granular_kappa(self):

        return 1

    def state_fraction(self, state_df):

        state_ = pd.merge(state_df, self.census, on=['vertex', 'age'], suffixes=('_state', '_census'))
        state_['prop_state'] = state_['N_state'] / state_['N_census']

        return state_

    def update_catchment(self, original, new, travel_df):

        update = pd.merge(original, new, right_on='vertex', left_on='vertex_src')
        update_ = pd.merge(update, travel_df, right_on='vertex_src', left_on='vertex_src',
                           suffixes=('_state', '_travel'))

        return update_

    def change_travel_inf(self, travel_df, hosp_catchment, symp_catchment, kappa):
        """Set of rules to
        - keep infected, symptomatic compartments home
        - send hospitalized compartment to ZCTAs corresponding to hospital catchments
        - remove the deceased from the contact patterns
        """

        Ih = self.state.loc[dict(compt='Ih')].to_dataframe()
        Iy = self.state.loc[dict(compt='Iy')].to_dataframe()
        D = self.state.loc[dict(compt='D')].to_dataframe()

        hosp_fraction = self.state_fraction(Ih)
        symp_fraction = self.state_fraction(Iy)
        dead_fraction = self.state_fraction(D)

        ## COMBINE STATES
        hosp_symp = pd.merge(
            hosp_fraction, symp_fraction, on=['vertex', 'age'], how='outer', suffixes=['_hosp', '_symp']
        )
        all_fractions = pd.merge(hosp_symp, dead_fraction, on=['vertex', 'age'], how='outer', suffixes=['', '_dead'])
        all_fractions['prop_travel'] = all_fractions['prop_state_hosp'] + all_fractions['prop_state_symp'] + all_fractions['prop_state_dead']

        ## INTERNAL SANITY CHECK
        total_fraction = (all_fractions['prop_travel'] + all_fractions['prop_state_hosp'] + all_fractions['prop_state_symp'] + all_fractions['prop_state_dead'])
        assert max(total_fraction) == 1.

        ## UNINFECTED
        # fraction with NO change in travel (implicitly removes symptomatic, hospitalized, and deceased)
        update_travel = pd.merge(travel_df, all_fractions, left_on=['vertex_src', 'age'], right_on=['vertex', 'age'])
        update_travel['n_updated'] = update_travel['n'] * all_fractions['prop_travel']
        update_travel = update_travel[['vertex_src', 'vertex_dest', 'age', 'n_updated']].rename(
            columns={'n_updated': 'n'}
        )

        ## HOSPITALS
        # fraction with change in travel
        """At this point in development, we want to analyze hospitalization post-hoc
        So we have defined hospital catchments, but we don't have hospitals in the contact matrix and we don't
        have a baseline census of people in hospitals. """
        #hosp_fraction_ = self.update_catchment(original=hosp_fraction, new=hosp_catchment, travel_df=travel_df)
        #hosp_fraction_['n_travel_hosp'] = hosp_fraction_['n_travel'] * hosp_fraction_['prop_state']
        #to_hosp = hosp_fraction_[['vertex_src', 'vertex_hosp', 'age', 'n_travel_hosp']].rename(
        #    columns={'vertex_hosp': 'vertex_dest', 'n_travel_hosp': 'n'}
        #)

        ## SYMPTOMATIC INFECTIONS
        # fraction with change in travel
        symp_fraction_ = self.update_catchment(original=symp_fraction, new=symp_catchment, travel_df=travel_df)
        symp_fraction_['n_stay_home'] = symp_fraction_['n_travel'] * symp_fraction_['prop_state']
        stay_home = symp_fraction_[['vertex_src', 'vertex_hosp', 'age', 'n_stay_home']].rename(
            columns={'vertex_hosp': 'vertex_dest', 'n_stay_home': 'n'}
        )

        ## FINALIZE
        final_travel = pd.concat([update_travel, stay_home])
        final_travel = final_travel.groupby(['vertex_src', 'vertex_dest', 'age'])

        return final_travel