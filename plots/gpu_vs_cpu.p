# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cpu_vs_gpu.png'
# The graphic title
set title 'OpenMP: CPU vs GPU Runtime'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Runtime (log(s))"
set key top
plot "./data/cpu.dat" using ((($1*$1*$1*8*3)/1e6)):2 with linespoints title "CPU (24 Threads)", \
     "./data/omp_single.dat" using  ((($1*$1*$1*8*3)/1e6)):2 with linespoints title "GPU, Data-Transfer", \
     "./data/omp_single.dat" using  ((($1*$1*$1*8*3)/1e6)):($2-$3) with linespoints title "GPU, no Data-Transfer"
