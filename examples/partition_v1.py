from episimlab.models import PartitionFromTravel

def main():
    model = PartitionFromTravel()
    model.run(input_vars={
        # update default input vars
        'sto_toggle': 0,
    })
    final_state = model.out_ds['compt_model__state']
    print(final_state.coords)
    model.plot()


if __name__ == '__main__':
	main()