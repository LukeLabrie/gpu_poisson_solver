# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'mpi_vs_cuda_band_2.png'
# The graphic title
set title 'MPI vs CUDA: Quadruple vs Dual GPU'
#plot the graphic
set logscale x
#set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Bandwidth (GB/s)"
set key top
set key font ",10"
set yrange [400:5200]
#set xrange [10:25000]
plot "./data/cuda_double.dat" using  ((3*8*$1*$1*$1)/1e6):5 with linespoints title "CUDA Dual, Data Transfer", \
     "./data/cuda_double.dat" using  ((3*8*$1*$1*$1)/1e6):6 with linespoints title "CUDA Dual, No Data Transfer", \
     "./data/cuda_single_dummy.dat" using  ((3*8*$1*$1*$1)/1e6):(2*$7)  with linespoints title "2 X NVIDIA V100 Rating", \
     "./data/mpi_nccl.dat" using  ((3*8*$1*$1*$1)/1e6):5 with linespoints title "MPI+NCCL (4 GPU, Data Transfer)", \
     "./data/mpi_nccl.dat" using  ((3*8*$1*$1*$1)/1e6):6 with linespoints title "MPI+NCCL (4 GPU, no Data Transfer)", \
     "./data/cuda_single_dummy.dat" using  ((3*8*$1*$1*$1)/1e6):(4*$7) with linespoints title "4 X NVIDIA V100 Rating"


