# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'single_vs_double.png'
# The graphic title
set title ''
#plot the graphic
set logscale x
set logscale y
set xlabel "N"
set ylabel "Runtime (s)"
set key bottom
plot "./data/split_cuda_double.dat" using  1:2 with linespoints title "Double", \
     "./data/split_cuda_single.dat" using 1:2 with linespoints title "Single"
