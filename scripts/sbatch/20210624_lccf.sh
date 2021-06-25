#!/bin/bash

#----------------------------------------------------
# Episimlab Benchmarks on Frontera for Meyers LCCF Proposal
# Last updated Ethan HO 6/24/2021
#----------------------------------------------------

#SBATCH -J esl_lccf           # Job name
#SBATCH -o logs/esl_lccf.o%j       # Name of stdout output file
#SBATCH -e logs/esl_lccf.e%j       # Name of stderr error file
#SBATCH -p development          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)

# Any other commands must follow all #SBATCH directives...
module load python3/3.7.0 gsl/2.6
module list
which python3
python3 --version

# NOTE: build cython modules if not built. Only need to do this once,
# so will not include in this script. Build using:
# python3 setup.py build_ext --inplace
# or 
# CC=$CCOMPILER make cython

# venv with python dependencies if not installed
pwd
VENV=./virpy
if [ ! -d $VENV ]; then
	python3 -m venv $VENV
fi

echo "Activating Python3 virtual env..."
source $VENV/bin/activate
python3 -m pip install -r requirements.txt
which python3
python3 --version
python3 -m pip freeze

# Launch script
OPTS=""
OPTS="$OPTS --config-fp scripts/20210625_lccf.yaml"
OPTS="$OPTS --travel-fp data/lccf/travel0.csv"
OPTS="$OPTS --contacts-fp data/lccf/contacts0.csv"
OPTS="$OPTS --census-counts-csv data/lccf/census0.csv"
PYTHONPATH='.' python3 scripts/20210623_lccf.py $OPTS
