# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cpu_vs_gpu_band.png'
# The graphic title
set title 'OpenMP: CPU vs GPU Bandwidth'
#plot the graphic
set logscale x
#set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Bandwidth (GB/s)"
set key top
#set yrange [0:200]
plot "./data/cpu.dat" using ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2)) with linespoints title "CPU (24 Threads)", \
     "./data/omp_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2)) with linespoints title "GPU, Data-Transfer", \
     "./data/omp_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3))) with linespoints title "GPU, no Data-Transfer"



