# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cuda_vs_omp_band.png'
# The graphic title
set title 'Single GPU: OpenMP vs CUDA'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Bandwidth (GB/s)"
set key bottom
set yr [5:1500]
plot "./data/omp_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "OpenMP", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "CUDA, Data-Transfer", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "CUDA, no Data-Transfer", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):7  with linespoints title "NVIDIA V100 Rating"
