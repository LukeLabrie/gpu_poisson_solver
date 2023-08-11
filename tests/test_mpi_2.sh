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
#BSUB -W 1:30
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"
#BSUB -o output_%J.out
#BSUB -e output_%J.err

mpirun -np 2 -pernode --bind-to none ./poisson_solver 256 1000 0.00 1 6 NONE | grep -v CPU >> /zhome/f2/b/166585/cuda/project/data/mpi_test$LOGEXT