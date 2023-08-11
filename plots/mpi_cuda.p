# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'mpi_vs_cuda.png'
# The graphic title
set title 'MPI vs CUDA: Quadruple vs Dual GPU'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Runtime (log(s))"
set key bottom
set key font ",10"
plot "./data/cuda_double.dat" using  ((3*8*$1*$1*$1)/1e6):2 with linespoints title "CUDA Dual, Data Transfer", \
     "./data/mpi_nccl.dat" using  ((3*8*$1*$1*$1)/1e6):2 with linespoints title "MPI+NCCL (4 GPU, Data Transfer)", \
     "./data/mpi_nccl.dat" using  ((3*8*$1*$1*$1)/1e6):($2-$3) with linespoints title "MPI+NCCL (4 GPU, no Data Transfer)"


