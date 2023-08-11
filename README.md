The driver executable can be called as follows 

```
./poisson_solver N K T_0 output_type [file_suffix] [threads]
```

`N`: Problem size. For single-GPU solvers, this needs to be a multiple of 16. For the dual-GPU CUDA solver, this needs to be a multiple of 32. 

`K`: Number of iterations.

`T_0`: Starting temeprature of inner points on the domain, in Kelvin. 


`output_type`: \
    0 = No output \
    1 = Performance metrics printed as [N] [wall time] [data transfer time (s)] [memory (MB)] [bandwidth (data transfer, GB/s)] [bandwidth (no data transfer, GB/s)] [time spent in kernel (s, not always measured)] [bandwidth based on kernel time (s, not always measured)] \
    3 = Write binary dump (.bin) \
    4 = Write .vtk file

    
`file_suffix`: Suffix to add to .vtk file


`threads`: Threads for CPU versions
