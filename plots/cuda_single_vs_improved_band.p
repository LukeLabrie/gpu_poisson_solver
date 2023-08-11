# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cuda_vs_omp_improved_band.png'
# The graphic title
set title 'CUDA: Initial vs Improved (CUDA 2)'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Bandwidth (GB/s)"
set key bottom
set yr [5:1500]
plot "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "CUDA, Data-Transfer", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "CUDA, no Data-Transfer", \
     "./data/cuda_single_improved.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "CUDA 2, Data-Transfer", \
     "./data/cuda_single_improved.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "CUDA 2, no Data-Transfer", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):9  with linespoints title "NVIDIA V100 Rating"
