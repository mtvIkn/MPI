#!/bin/bash
#
#SBATCH --nodes = 4
#SBATCH --tasks-per-node = 32 
#SBATCH --time = 05:00:00

mpicc.mpich MPI.cpp -o MPI.out
#chmod u+x MPI.out

mpirun.mpich ./MPI.out