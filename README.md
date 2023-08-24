# GPU Poisson Solver

## Background

This solver solves the Poisson equation in three dimensions for a specific set of boundary conditions. The poisson equation can be expressed as 

```math
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} = -f(x,y,z), \quad (x,y,x) \in \Omega
```

Different methods will be compared for solvers operating on a cubical domain $\Omega$ defined as

```math
\Omega = \{(x, y, z) : |x| \leq 1, |y| \leq 1, |z| \leq 1\}    
```

with boundary conditions 

```math
\begin{align*}
u(x, 1, z) &= 20, u(x, -1, z) = 0,  -1 \leq x, z \leq 1 \\
u(1, y, z) &= u(-1, y, z) = 20,  -1 \leq y, z \leq 1 \\
u(x, y, -1) &= u(x, y, 1) = 20,  -1 \leq x, y \leq 1.
\end{align*}
```

The purpose of the solver is to compare performance across different parallelization methods and includes parallel methods for:
- CPU
- Single-GPU via OpenMP
- Dual-GPU via OpenMP
- Single-GPU via CUDA
- Dual-GPU via CUDA
- Four-GPU (two nodes, each with two GPUs), via CUDA, OpenMPI, and NCCL 
   
A full writeup is available in the [report](report.pdf).

## Requirements 

The project is compiled with the `mpic++` compiler, an OpenMPI C++ wrapper compiler. The underlying compiler is set to `nvc++`, NVIDIA's compiler for their GPUS. This can be done by setting the environment variable `OMPI_CXX` to `nvc++` via

```
export OMPI_CXX=nvc++
```

Other requirements include:
- CUDA
- OpenMPI
- OpenMP
- NCCL 

## Executable 

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
