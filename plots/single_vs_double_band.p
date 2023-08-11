# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'single_vs_double_band.png'
# The graphic title
set title 'OpenMP: Single vs Dual GPU'
#plot the graphic
set logscale x
#set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Bandwidth (GB/s)"
set key top
set yrange [0:300]
plot "./data/omp_single.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "Single GPU", \
     "./data/omp_double.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "Dual GPU, Data-Transfer", \
     "./data/omp_double.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "Dual GPU, no Data-Transfer"
     