# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'single_vs_double.png'
# The graphic title
set title 'OpenMP: Single vs Dual GPU'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "Runtime (log(s))"
set key top
plot "./data/omp_single.dat" using  ((($1*$1*$1*8*3)/1e6)):2 with linespoints title "Single GPU", \
     "./data/omp_double.dat" using  ((($1*$1*$1*8*3)/1e6)):2 with linespoints title "Dual GPU, Data-Transfer", \
     "./data/omp_double.dat" using  ((($1*$1*$1*8*3)/1e6)):($2-$3) with linespoints title "Dual GPU, no Data-Transfer"