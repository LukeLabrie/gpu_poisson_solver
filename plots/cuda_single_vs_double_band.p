# Set the output to a png file
set terminal png size 500,500
# The file we'll write to
set output 'cuda_single_vs_double_band.png'
# The graphic title
set title 'Single vs Dual GPU: CUDA'
#plot the graphic
set logscale x
set logscale y
set xlabel "Memory Footprint (log(MB))"
set ylabel "log(GB/s)"
set key bottom
set yr [5:3000]
plot "./data/cuda_single_improved.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "Single, No Data Transfer", \
     "./data/cuda_double.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*$2))  with linespoints title "Dual, Data Transfer", \
     "./data/cuda_double.dat" using  ((3*8*$1*$1*$1)/1e6):((3*8*$1*$1*$1)/(1e6*($2-$3)))  with linespoints title "Dual, no Data Transfer", \
     "./data/cuda_single_dummy.dat" using  ((3*8*$1*$1*$1)/1e6):7  with linespoints title "NVIDIA V100 Rating", \
     "./data/cuda_single_dummy.dat" using  ((3*8*$1*$1*$1)/1e6):(2*$7)  with linespoints title "2 X NVIDIA V100 Rating"
