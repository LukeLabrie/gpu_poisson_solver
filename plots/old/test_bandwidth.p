# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cuda_vs_omp.png'
# The graphic title
set title ''
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log kB)"
set ylabel "Runtime (s)"
set key bottom
plot "./data/omp_single.data" using  ((($1*$1*$1*8*3)/1e3)/$3) with linespoints title "OpenMP", \
     "./data/cuda_single.data" using ((($1*$1*$1*8*3)/1e3)/$3) with linespoints title "CUDA"
