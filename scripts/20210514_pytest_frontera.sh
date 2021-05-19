#!/bin/bash

#SBATCH -J episimlab-pytest           # Job name
#SBATCH -o logs/esl-pytest.o%j       # Name of stdout output file
#SBATCH -e logs/esl-pytest.e%j       # Name of stderr error file
#SBATCH -p development          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A COVID19-Portal       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=eho@tacc.utexas.edu

# ------------------------------------------------------------------------------

module load python3/3.9.2 gsl/2.6
source $HOME/.poetry/env

date
time poetry run pytest