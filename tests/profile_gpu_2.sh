#!/bin/sh

### -- set the job Name --
#BSUB -J MPI_NCCL
### –- specify queue --
#BSUB -q mementogpu
### -- ask for number of cores (default: 1) --
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[ptile=4]"
### -- set walltime limit: hh:mm --
#BSUB -W 0:05
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
#BSUB -o output_%J.out
#BSUB -e output_%J.err

mpirun -np 2 -pernode --bind-to none nsys profile -t cuda,nvtx,mpi,openmp ./poisson_solver 512 1000 0.000 1 6 test_mpi 