from episimlab.models import Vaccine

def main():
    model = Vaccine()
    model.run(input_vars={
        # update default input vars
        'sto_toggle': 0,
        'contact_da_fp': 'data/20200311_contact_matrix.nc',
    })
    final_state = model.out_ds['compt_model__state']
    print(final_state.coords)
    model.plot()


if __name__ == '__main__':
	main()
