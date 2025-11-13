#!/bin/bash
# run_examples.sh
if [ -z "$1" ]; then
  echo "Usage: ./run_examples.sh <nprocs> <nqubits>"
  exit 1
fi
NP=$1
NQ=${2:-3}
mpirun -n $NP ./qsim $NQ
