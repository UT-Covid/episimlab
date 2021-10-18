from episimlab.models import Vaccine
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

def main():
    model = Vaccine()
    model.run(input_vars={
        # update default input vars
        'sto_toggle': 0,
        'contacts_fp': '/Users/kpierce/COVID19/SchoolCatchmentDemo/mock_polymod_contacts.csv',
        'travel_fp': '/Users/kpierce/COVID19/SchoolCatchmentDemo/spring_semester_2020_travel.csv',
        'initial_state_df': '/Users/kpierce/COVID19/SchoolCatchmentDemo/mock_initial_state_census.csv'
    })
    final_state = model.out_ds['compt_model__state']
    print(final_state.coords)
    model.plot()


if __name__ == '__main__':
	main()
