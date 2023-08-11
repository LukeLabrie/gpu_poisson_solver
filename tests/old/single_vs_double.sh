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

LOGEXT=.dat

sizes=(16 32 64 128 160 192 256 512 640)

for ((i=0;i<10;i++))
do
./poisson_solver ${sizes[i]} 500 0.000 0.00 1 4 | grep -v CPU >> /zhome/f2/b/166585/cuda/project/data/split_cuda_single$LOGEXT
done


for ((i=0;i<10;i++))
do
./poisson_solver ${sizes[i]} 500 0.000 0.00 1 5 | grep -v CPU >> /zhome/f2/b/166585/cuda/project/data/split_cuda_double$LOGEXT
done
