# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cuda_vs_omp_improved.png'
# The graphic title
set title 'CUDA: Initial vs Improved (CUDA 2)'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Runtime (log(s)) "
set key bottom
plot "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):2  with linespoints title "CUDA, Data-Transfer", \
     "./data/cuda_single.dat" using  ((3*8*$1*$1*$1)/1e6):($2-$3)  with linespoints title "CUDA, No Data-Transfer", \
     "./data/cuda_single_improved.dat" using  ((3*8*$1*$1*$1)/1e6):2  with linespoints title "CUDA 2, Data-Transfer", \
     "./data/cuda_single_improved.dat" using  ((3*8*$1*$1*$1)/1e6):($2-$3)  with linespoints title "CUDA 2, No Data-Transfer"