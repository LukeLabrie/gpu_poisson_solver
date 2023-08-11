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
#BSUB -R "rusage[mem=32GB]"
#BSUB -o output_%J.out
#BSUB -e output_%J.err

LOGEXT=.dat

sizes=(16 32 64 128 256 512 576 640 704 768 800 832)

for ((i=0;i<12;i++))
do
./poisson_solver ${sizes[i]} 1000 0.000 1 6 | grep -v CPU >> /zhome/f2/b/166585/cuda/project/data/cuda_double$LOGEXT
done
