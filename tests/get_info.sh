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

rm get_info.txt
module purge
module load nccl/2.17.1-1-cuda-12.1
module load mpi/4.1.4-gcc-12.2.0-binutils-2.39

nvidia-smi > gpu_info.txt
echo >> gpu_info.txt

echo "All Nodes:" >> gpu_info.txt
nodestat mementogpu >> gpu_info.txt
echo >> gpu_info.txt

echo "Current Node:" >> gpu_info.txt
hostname >> gpu_info.txt
echo >> gpu_info.txt

echo "CPU Info:" >> gpu_info.txt
lscpu >> gpu_info.txt

./query_gpu