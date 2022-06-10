import logging
import pandas as pd
import numpy as np
import xarray as xr
import xsimlab as xs
from ..utils import group_dict_by_var, get_int_per_day, fix_coord_dtypes, get_var_dims
from .travel_pat import TravelPatFromCSV

@xs.process
class BehaviorChange(TravelPatFromCSV):

    state = xs.global_ref('state', intent='in')
    hosp_catchment_fp = xs.variable(
        static=True, intent='in', description="path to a CSV file containing hospital catchments"
    )

    @xs.runtime(args=('step_end', 'step_start'))
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

        kappa = self.get_granular_kappa()

        updated_df = self.change_travel_inf(travel_df, kappa)

        self.travel_pat = self.get_travel_da(updated_df)

    def get_granular_kappa(self):

        return 1

    def state_fraction(self, state_df, initial_state):

        state_ = pd.merge(state_df, initial_state, on=['vertex', 'age'], suffixes=('_state', '_census'))
        state_['prop_state'] = state_['n_state'] / state_['n_census']

        return state_

    def change_travel_inf(self, travel_df, kappa):
        """Set of rules to
        - keep infected, symptomatic compartments home
        - send hospitalized compartment to ZCTAs corresponding to hospital catchments
        - remove the deceased from the contact patterns
        """

        # grab the initial state by summing across all compartments and risk groups
        # this works because there are no births or immigrations, and death (removal) is tracked
        initial_state = self.state.to_dataframe(name='n')
        initial_state = initial_state.groupby(['vertex', 'age'])['n'].sum().reset_index()

        Ih = self.state.loc[dict(compt='Ih')].to_dataframe(name='n')
        Iy = self.state.loc[dict(compt='Iy')].to_dataframe(name='n')
        D = self.state.loc[dict(compt='D')].to_dataframe(name='n')

        hosp_fraction = self.state_fraction(Ih, initial_state).rename(columns={'prop_state': 'prop_state_hosp'})
        symp_fraction = self.state_fraction(Iy, initial_state).rename(columns={'prop_state': 'prop_state_symp'})
        dead_fraction = self.state_fraction(D, initial_state).rename(columns={'prop_state': 'prop_state_dead'})

        ## COMBINE STATES
        hosp_symp = pd.merge(hosp_fraction, symp_fraction, on=['vertex', 'age'], how='outer')
        all_fractions = pd.merge(hosp_symp, dead_fraction, on=['vertex', 'age'], how='outer')
        all_fractions['prop_no_travel'] = all_fractions['prop_state_hosp'] + all_fractions['prop_state_symp'] + all_fractions['prop_state_dead']

        ## UNINFECTED
        # fraction with NO change in travel (implicitly removes symptomatic, hospitalized, and deceased)
        update_travel = pd.merge(travel_df, all_fractions, left_on=['source', 'age'], right_on=['vertex', 'age'])
        update_travel['n_updated'] = update_travel['n'] * (1 - update_travel['prop_no_travel'])
        update_travel_ = update_travel[['source', 'destination', 'age', 'n_updated']].rename(
            columns={'n_updated': 'n'}
        )

        ## HOSPITALS
        # fraction with change in travel
        """At this point in development, we want to analyze hospitalization post-hoc
        So we have defined hospital catchments, but we don't have hospitals in the contact matrix and we don't
        have a baseline census of people in hospitals. """

        ## SYMPTOMATIC INFECTIONS
        # fraction with change in travel
        update_travel['n_stay_home'] = update_travel['n'] * update_travel['prop_state_symp']
        update_symp = update_travel[['source', 'age', 'n_stay_home']]
        update_symp['destination'] = update_symp['source']
        update_symp = update_symp.rename(columns={'n_stay_home': 'n'})

        ## FINALIZE
        final_travel = pd.concat([update_travel_, update_symp])
        final_travel = final_travel.groupby(['source', 'destination', 'age'])['n'].sum().reset_index()

        ## INTERNAL SANITY CHECK
        final_travel_vertex_total = final_travel.groupby(['source', 'age'])['n'].sum().reset_index()
        compare = pd.merge(initial_state, final_travel_vertex_total, left_on=['vertex'], right_on=['source'])
        diff = compare['n_x'] - compare['n_y']
        np.testing.assert_almost_equal(max(diff), 0)
        np.testing.assert_almost_equal(min(diff), 0)

        return final_travel