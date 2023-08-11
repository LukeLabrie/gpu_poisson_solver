/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <stdbool.h>
#include "/appl/nccl/2.17.1-1-cuda-12.1/include/nccl.h"
#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>
#include "alloc3d_cuda.h"
#include "print.h" 

/*

MACROS

*/
#define IDX3D(i, j, k, n) ((k) + (n)*(j) + (i)*(n)*(n))
#define IDX2D(j, k, n) ((k) + (n)*(j))

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

/*

KERNELS

*/
__global__
void poisson_single(double *u_d, double *v_d, double *f_d, double div, double del2, int n) {

    /*
    Single GPU kernel
    */
   
    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    bool boundary = (i == (n-1) || j == (n-1) || k == (n-1) || i == 0 || j == 0 || k == 0);
    if (!boundary) {
        int global = IDX3D(i,j,k,n);

        // naming convention from perspective of xz plane
        double down = i > 1 ? u_d[IDX3D(i-1,j,k,n)] : 20.0;
        double up = i < n-2 ? u_d[IDX3D(i+1,j,k,n)] : 20.0;
        double back = j > 1 ? u_d[IDX3D(i,j-1,k,n)] : 0.0;
        double forward = j < n-2 ? u_d[IDX3D(i,j+1,k,n)] : 20.0;
        double right = k < n-2 ? u_d[IDX3D(i,j,k+1,n)] : 20.0;
        double left = k > 1 ? u_d[IDX3D(i,j,k-1,n)] : 20.0;

        // make update
        v_d[global] = div*(down+up+left+right+forward+back+del2*f_d[global]);
    }
}

__global__
void poisson_single_2(double *u_d, double *v_d, double *f_d, double div, double del2, int n) {

    /*
    Single GPU kernel, no branching 
    */

    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x+1;
    int j = blockIdx.y * blockDim.y + threadIdx.y+1;
    int i = blockIdx.z * blockDim.z + threadIdx.z+1;

    int global = IDX3D(i,j,k,n);

    // naming convention from perspective of xz plane
    double right = u_d[IDX3D(i,j+1,k,n)];
    double forward = u_d[IDX3D(i,j,k+1,n)];
    double up = u_d[IDX3D(i+1,j,k,n)];
    double back = u_d[IDX3D(i,j,k-1,n)];
    double left = u_d[IDX3D(i,j-1,k,n)];
    double down = u_d[IDX3D(i-1,j,k,n)];

    // make update
    v_d[global] = div*(down+up+left+right+forward+back+del2*f_d[global]);

}

__global__
void poisson_single_3(double *u_down, double *u, double *u_up, double *v, double *f, double div, double del2, int n) {

    /*
    Single GPU kernel, layer-by-layer
    */

    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x+1;
    int j = blockIdx.y * blockDim.y + threadIdx.y+1;
    int global = IDX2D(j,k,n);

    // naming convention from perspective of xz plane
    double down = u_down[global];
    double up = u_up[global];
    double left = u[IDX2D(j,k-1,n)];
    double right = u[IDX2D(j,k+1,n)];
    double forward = u[IDX2D(j+1,k,n)];
    double back = u[IDX2D(j-1,k,n)];

    // make update
    v[global] = div*(down+up+left+right+forward+back+del2*f[global]);
}

#define BLOCK_SIZE 8
#define SHARED_SIZE (BLOCK_SIZE + 2)  // adding halo cells

__global__
void poisson_single_4(double *u_d, double *v_d, double *f_d, double div, double del2, int n) {

    /*
    Single GPU kernel, shared memory implementation
    */

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int z = bz * blockDim.z + tz;
    bool boundary = (x == (n-1) || y == (n-1) || z == (n-1) || x == 0 || y == 0 || z == 0);

    __shared__ double u_shared[SHARED_SIZE][SHARED_SIZE][SHARED_SIZE];
    __shared__ double f_shared[SHARED_SIZE][SHARED_SIZE][SHARED_SIZE];

    // Load input elements into shared memory
    if (x < n && y < n && z < n) {
        u_shared[tx+1][ty+1][tz+1] = u_d[IDX3D(x,y,z,n)];
        f_shared[tx+1][ty+1][tz+1] = f_d[IDX3D(x,y,z,n)];
    }

    // Load halo elements
    if (tx == 0 && x > 0) u_shared[0][ty+1][tz+1] = u_d[IDX3D(x-1,y,z,n)];
    if (ty == 0 && y > 0) u_shared[tx+1][0][tz+1] = u_d[IDX3D(x,y-1,z,n)];
    if (tz == 0 && z > 0) u_shared[tx+1][ty+1][0] = u_d[IDX3D(x,y,z-1,n)];
    if (tx == BLOCK_SIZE-1 && x < n-1) u_shared[BLOCK_SIZE+1][ty+1][tz+1] = u_d[IDX3D(x+1,y,z,n)];
    if (ty == BLOCK_SIZE-1 && y < n-1) u_shared[tx+1][BLOCK_SIZE+1][tz+1] = u_d[IDX3D(x,y+1,z,n)];
    if (tz == BLOCK_SIZE-1 && z < n-1) u_shared[tx+1][ty+1][BLOCK_SIZE+1] = u_d[IDX3D(x,y,z+1,n)];

    __syncthreads();

    if (!boundary) {
        double down = u_shared[tx][ty+1][tz+1];
        double up = u_shared[tx+2][ty+1][tz+1];
        double left = u_shared[tx+1][ty][tz+1];
        double right = u_shared[tx+1][ty+2][tz+1];
        double forward = u_shared[tx+1][ty+1][tz+2];
        double back = u_shared[tx+1][ty+1][tz];
        v_d[IDX3D(x,y,z,n)] = div*(down+up+left+right+forward+back+del2*f_shared[tx+1][ty+1][tz+1]);
    }
}

__global__
void poisson_single_5(double *u_d, double *v_d, double div, double del2, int n) {

    /*
    Single GPU kernel
    */
   
    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    // variables 
    double x, y, z; 
    bool rad, boundary;

    // location
    x = -1.0 + (2.0/(double)n)*(double)k;
    y = -1.0 + (2.0/(double)n)*(double)j;
    z = -1.0 + (2.0/(double)n)*(double)i;
    rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);
    boundary = (i == (n-1) || j == (n-1) || k == (n-1) || i == 0 || j == 0 || k == 0);
    if (!boundary) {
        int global = IDX3D(i,j,k,n);

        // naming convention from perspective of xz plane
        double right = k < n-2 ? u_d[IDX3D(i,j,k+1,n)] : 20.0;
        double forward = j < n-2 ? u_d[IDX3D(i,j+1,k,n)] : 20.0;
        double up = i < n-2 ? u_d[IDX3D(i+1,j,k,n)] : 20.0;
        double back = j > 1 ? u_d[IDX3D(i,j-1,k,n)] : 0.0;
        double left = k > 1 ? u_d[IDX3D(i,j,k-1,n)] : 20.0;
        double down = i > 1 ? u_d[IDX3D(i-1,j,k,n)] : 20.0;

        // make update
        if (rad) v_d[global] = div*(down+up+left+right+forward+back+del2*200.00);
        else v_d[global] = div*(down+up+left+right+forward+back);
    }
}

__global__
void poisson_split(double *u_d1, // matricies for 1st GPU
                   double *v_d, 
                   double *u_d2, // matricies for 2nd GPU
                   double div,   // coefficients for stencil formula
                   double del2,  
                   int n,        // dimensions of XY region   
                   int dev      // device number
                   ) {

    /*
    Double GPU kernel, remove F accesses, 
    can also reduce number of arguments, just change update 
    based on device number
    */
    
    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    // variables 
    int half = n/2;
    int global = IDX3D(i,j,k,n);
    double x, y, z; 
    bool rad, boundary, i_boundary;

    // location
    x = -1.0 + (2.0/(double)n)*(double)k;
    y = -1.0 + (2.0/(double)n)*(double)j;
    z = -1.0 + (2.0/(double)n)*(double)i + 1.0*dev;
    rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);
    i_boundary = ((dev == 0) && (i == 0)) || ((dev == 1) && (i == (half-1)));
    boundary = (i_boundary || j == (n-1) || k == (n-1) || j == 0 || k == 0);

    // naming convention from perspective of xz plane
    if (!boundary) {
        double down, up, back, forward, left, right;
        if (dev == 0) {
            right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
            forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
            up = i < (half-1) ? u_d1[IDX3D(i+1,j,k,n)] : u_d2[IDX3D(0,j,k,n)];
            back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
            left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
            down = i > 1 ? u_d1[IDX3D(i-1,j,k,n)] : 20.0;
        }
        else if (dev == 1) {
            right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
            forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
            up = i < (half-2) ? u_d1[IDX3D(i+1,j,k,n)] : 20.0;
            back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
            left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
            down = i > 0 ? u_d1[IDX3D(i-1,j,k,n)] : u_d2[IDX3D(half-1,j,k,n)];
        }
        // make update
        if (rad) v_d[global] = div*(down+up+left+right+forward+back+del2*200.00);
        else v_d[global] = div*(down+up+left+right+forward+back);
    }
}

__global__
void poisson_split_MPI(double *u_d1, // matricies for 1st GPU
                       double *v_d, 
                       double *u_d2, // matricies for 2nd GPU
                       double div,   // coefficients for stencil formula
                       double del2,  
                       int n,        // dimensions of XY region   
                       int dev,      // device number
                       int rank      // MPI rank 
                   ) {

    /*
    Double GPU kernel for MPI implementation (will solve 1/2 of domain)
    */
    
    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int global = IDX3D(i,j,k,n);
    int half = n/4; // notice factor of 4 here, as opposed to 2 for poisson_split kernel

    // domain variables
    bool z_boundaries, boundary, rad, out;
    double x, y, z;
    out = (i > half-1) || (i < 0);
    z_boundaries = ((dev == 0) && (i == 0)) || ((dev == 1) && (i == (half-1)));
    boundary = (z_boundaries || j >= (n-1) || k >= (n-1) || j <= 0 || k <= 0 || out);

    // location
    x = -1.0 + (2.0/(double)n)*(double)k;
    y = -1.0 + (2.0/(double)n)*(double)j;
    z = -1.0 + (2.0/(double)n)*(double)i + 0.5*dev + 1.0*rank;
    rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);

    // naming convention from perspective of xz plane
    // if statements go from bottom to top of domain
    if (!boundary) {
    double up, down, left, right, forward, back;
    // bottom quarter
    if (dev == 0 && rank == 0) {
        right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
        forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
        up = i < (half-1) ? u_d1[IDX3D(i+1,j,k,n)] : u_d2[IDX3D(0,j,k,n)];
        back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
        left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
        down = i > 1 ? u_d1[IDX3D(i-1,j,k,n)] : 20.0;
    }
    // 2nd quarter
    else if (dev == 1 && rank == 0) {
        right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
        forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
        up = u_d1[IDX3D(i+1,j,k,n)];  // potential node boundary
        back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
        left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
        down = i > 0 ? u_d1[IDX3D(i-1,j,k,n)] : u_d2[IDX3D(half-1,j,k,n)];
    }
    // 3rd quarter 
    else if (dev == 0 && rank == 1) {
        right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
        forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
        up = i < (half-1) ? u_d1[IDX3D(i+1,j,k,n)] : u_d2[IDX3D(0,j,k,n)];
        back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
        left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
        down = u_d1[IDX3D(i-1,j,k,n)]; // potential node boundary
    }    
    // top quarter
    else if (dev == 1 && rank == 1) {
        right = k < n-2 ? u_d1[IDX3D(i,j,k+1,n)] : 20.0;
        forward = j < n-2 ? u_d1[IDX3D(i,j+1,k,n)] : 20.0;
        up = i < (half-2) ? u_d1[IDX3D(i+1,j,k,n)] : 20.0;                     
        back = j > 1 ? u_d1[IDX3D(i,j-1,k,n)] : 0.0;
        left = k > 1 ? u_d1[IDX3D(i,j,k-1,n)] : 20.0;
        down = i > 0 ? u_d1[IDX3D(i-1,j,k,n)] : u_d2[IDX3D(half-1,j,k,n)];
    }
    // make update
    if (rad) v_d[global] = div*(down+up+left+right+forward+back+del2*200.00);
    else v_d[global] = div*(down+up+left+right+forward+back);
    }
}

__global__
void poisson_boundary(double *u_d, // local matricies 
                      double *v_d, 
                      double *u_d2, // boundary data (recvbuff)
                      double div,   // coefficients for stencil formula
                      double del2,  
                      int n,        // dimensions of XY region   
                      int rank      // MPI rank
                        ) {
    /*
    Boundary Kernel
    */

    // get global indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int global = 0;
    int half = n/4;
    if (rank == 0) global = IDX3D((half-1),j,k,n);
    else if (rank == 1) global = IDX3D(0,j,k,n);
    bool boundary, rad;
    boundary = (j == (n-1) || k == (n-1) || j == 0 || k == 0);

    // location
    double x, y, z;
    x = -1.0 + (2.0/(double)n)*(double)k;
    y = -1.0 + (2.0/(double)n)*(double)j;
    z = 0.0;
    rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);

    if (!boundary){
    double up, down, left, right, forward, back;
    if (rank == 0) {
        // naming convention from perspective of xz plane
        right = j < n-2 ? u_d[IDX3D(half-1,j+1,k,n)] : 20.0;
        forward = k < n-2 ? u_d[IDX3D(half-1,j,k+1,n)] : 20.0;
        back = k > 1 ? u_d[IDX3D(half-1,j,k-1,n)] : 20.0;
        left = j > 1 ? u_d[IDX3D(half-1,j-1,k,n)] : 0.0;
        down = u_d[IDX3D(half-2,j,k,n)];
        up = u_d2[IDX3D(0,j,k,n)];
    }
    else if (rank == 1) {
        // naming convention from perspective of xz plane
        right = j < n-2 ? u_d[IDX3D(0,j+1,k,n)] : 20.0;
        forward = k < n-2 ? u_d[IDX3D(0,j,k+1,n)] : 20.0;
        back = k > 1 ? u_d[IDX3D(0,j,k-1,n)] : 20.0;
        left = j > 1 ? u_d[IDX3D(0,j-1,k,n)] : 0.0;
        down = u_d2[IDX3D(0,j,k,n)];
        up = u_d[IDX3D(1,j,k,n)];

    }
    // make update
    if (rad) v_d[global] = div*(down+up+left+right+forward+back+del2*200.00);
    else v_d[global] = div*(down+up+left+right+forward+back);
    }   
}

/*

SOLVERS

*/
void jacobi_cuda_single(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {
    /*
    Single-GPU cuda implementation
    */

    // device pointers for matiriies
    double *U_d;
    double *V_d;
    double *F_d;

    // measure execution time
    double start_run = omp_get_wtime();

    // allocate
    const int sizeBytes = N*N*N*sizeof(double);
    CUDACHECK(cudaMalloc((void **)&U_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&V_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&F_d, sizeBytes));

    // populate
    cudaMemcpy(U_d, U[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);

    // stop data-transfer clock
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start_run;
    *dt = duration_dt;

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for kernel 
    dim3 blocks(8, 8, 16); // a 8x8x8 block of threads
    dim3 grid((N) / blocks.x, (N) / blocks.y, (N) / blocks.z); // enough blocks to cover the whole 3D domain
    int m = 0;

    // call kernel
    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    *kt = 0.0;
    double sk;
    while (m<K) {
        sk = omp_get_wtime();
        poisson_single<<<grid, blocks>>>(U_d,V_d,F_d,div,del2,N);
        cudaDeviceSynchronize();
        *kt += omp_get_wtime()-sk;
        m += 1;
        double *old = U_d;
        U_d = V_d;
        V_d = old;
    }

    // transfer data back to host and add to data transfer measurement
    double out = omp_get_wtime();
    cudaMemcpy(U[0][0], U_d, N*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // free device memory
    free_3d_cuda(U_d);
    free_3d_cuda(V_d);
    free_3d_cuda(F_d);

    // end execution time measurement
    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start_run;
}

void jacobi_cuda_single_2(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {

    /*
    Single-GPU cuda implementation, without any kernel branching 
    */

    // device pointers for matiriies
    double *U_d;
    double *V_d;
    double *F_d;

    // measure execution time
    double start_run = omp_get_wtime();

    // allocate
    const int sizeBytes = N*N*N*sizeof(double);
    CUDACHECK(cudaMalloc((void **)&U_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&V_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&F_d, sizeBytes));

    // populate
    cudaMemcpy(U_d, U[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);

    // stop data-transfer clock
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start_run;
    *dt = duration_dt;

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for kernel 
    dim3 blocks(8, 8, 16); // a 8x8x8 block of threads
    dim3 grid((N-2) / blocks.x, (N-2) / blocks.y, (N-2) / blocks.z); // enough blocks to cover the whole 3D domain
    int m = 0;

    // call kernel
    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    *kt = 0.0;
    double sk;
    while (m<K) {
        sk = omp_get_wtime();
        poisson_single_2<<<grid, blocks>>>(U_d,V_d,F_d,div,del2,N);
        cudaDeviceSynchronize();
        *kt += omp_get_wtime()-sk;
        m += 1;
        double *old = U_d;
        U_d = V_d;
        V_d = old;
    }

    // transfer data back to host and add to data transfer measurement
    double out = omp_get_wtime();
    cudaMemcpy(U[0][0], U_d, N*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // free device memory
    free_3d_cuda(U_d);
    free_3d_cuda(V_d);
    free_3d_cuda(F_d);

    // end execution time measurement
    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start_run;

}

void jacobi_cuda_single_3(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {

    /*
    Single-GPU cuda implementation: allocate each layer separately to try to reduce memory load on kernels
    */

    // measure execution time
    double start_run = omp_get_wtime();

    // device pointers for matiriies
    double *U_d[N];
    double *V_d[N];
    double *F_d[N];


    // allocate & populate for each layer 
    const int sizeBytes = N*N*sizeof(double);
    for (int l=0;l<N;l++) {
        // allocate 
        CUDACHECK(cudaMalloc(U_d+l, sizeBytes));
        CUDACHECK(cudaMalloc(V_d+l, sizeBytes));
        CUDACHECK(cudaMalloc(F_d+l, sizeBytes));

        // populate
        CUDACHECK(cudaMemcpyAsync(U_d[l], U[l][0], sizeBytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpyAsync(V_d[l], V[l][0], sizeBytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpyAsync(F_d[l], F[l][0], sizeBytes, cudaMemcpyHostToDevice));
    }

    // synchronize 
    CUDACHECK(cudaDeviceSynchronize());

    // stop data-transfer clock
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start_run;
    *dt = duration_dt;

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for kernel 
    dim3 blocks(16, 8, 4); // a 8x8x8 block of threads
    dim3 grid((N-2) / blocks.x, (N-2) / blocks.y, (N-2) / blocks.z); // enough blocks to cover the whole 3D domain
    int m = 0;

    // call kernel
    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);

    /*
    TRY A LOOP SENDING 3-LAYERED KERNELS
    */

   *kt = 0.00;
    while (m<K) {
        double start_k = omp_get_wtime();
        for (int j = 1; j<(N-1); j++) poisson_single_3<<<grid, blocks>>>(U_d[j-1],U_d[j],U_d[j+1],V_d[j],F_d[j],div,del2,N);
        CUDACHECK(cudaDeviceSynchronize());
        *kt += omp_get_wtime()-start_k;
        for (int j = 1; j<(N-1); j++) {
            double *old = U_d[j];
            U_d[j] = V_d[j];
            V_d[j] = old;
        }
        m += 1;
    }

    // transfer data back to host and add to data transfer measurement
    double out = omp_get_wtime();
    for (int l=1;l<N-1;l++) {

        // populate
        CUDACHECK(cudaMemcpyAsync(U[l][0], U_d[l], sizeBytes, cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpyAsync(V[l][0], U_d[l], sizeBytes, cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpyAsync(F[l][0], U_d[l], sizeBytes, cudaMemcpyDeviceToHost));
    }
    CUDACHECK(cudaDeviceSynchronize());
    
    // free device memory
    for (int i = 0; i < N; i++) {
        cudaSetDevice(i);
        free_3d_cuda(U_d[i]);
        free_3d_cuda(V_d[i]);
        free_3d_cuda(F_d[i]);
    }
    // end execution time measurement
    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start_run;
}

void jacobi_cuda_single_4(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {

    /*
    Single-GPU cuda implementation, shared memory
    */

    // device pointers for matiriies
    double *U_d;
    double *V_d;
    double *F_d;

    // measure execution time
    double start_run = omp_get_wtime();

    // allocate
    const int sizeBytes = N*N*N*sizeof(double);
    CUDACHECK(cudaMalloc((void **)&U_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&V_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&F_d, sizeBytes));

    // populate
    cudaMemcpy(U_d, U[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);

    // stop data-transfer clock
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start_run;
    *dt = duration_dt;

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for kernel 
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + blocks.x - 1) / blocks.x, (N + blocks.y - 1) / blocks.y, (N + blocks.z - 1) / blocks.z);
    int m = 0;

    // call kernel
    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    *kt = 0.0;
    double sk;
    while (m<K) {
        sk = omp_get_wtime();
        poisson_single_4<<<grid, blocks>>>(U_d,V_d,F_d,div,del2,N);
        CUDACHECK(cudaDeviceSynchronize());
        *kt += omp_get_wtime()-sk;
        m += 1;
        double *old = U_d;
        U_d = V_d;
        V_d = old;
    }

    // transfer data back to host and add to data transfer measurement
    double out = omp_get_wtime();
    cudaMemcpy(U[0][0], U_d, N*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // free device memory
    free_3d_cuda(U_d);
    free_3d_cuda(V_d);
    free_3d_cuda(F_d);

    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start_run;

}

void jacobi_cuda_single_5(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {
    /*
    Single-GPU cuda implementation, don't use a full array for F
    */

    // device pointers for matiriies
    double *U_d;
    double *V_d;

    // measure execution time
    double start_run = omp_get_wtime();

    // allocate
    const int sizeBytes = N*N*N*sizeof(double);
    CUDACHECK(cudaMalloc((void **)&U_d, sizeBytes));
    CUDACHECK(cudaMalloc((void **)&V_d, sizeBytes));

    // populate
    cudaMemcpy(U_d, U[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V[0][0], N*N*N*sizeof(double), cudaMemcpyHostToDevice);

    // stop data-transfer clock
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start_run;
    *dt = duration_dt;

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for kernel 
    dim3 blocks(8, 8, 16); // a 8x8x8 block of threads
    dim3 grid((N) / blocks.x, (N) / blocks.y, (N) / blocks.z); // enough blocks to cover the whole 3D domain
    int m = 0;

    // call kernel
    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    *kt = 0.0;
    double sk;
    while (m<K) {
        sk = omp_get_wtime();
        poisson_single_5<<<grid, blocks>>>(U_d,V_d,div,del2,N);
        cudaDeviceSynchronize();
        *kt += omp_get_wtime()-sk;
        m += 1;
        double *old = U_d;
        U_d = V_d;
        V_d = old;
    }

    // transfer data back to host and add to data transfer measurement
    double out = omp_get_wtime();
    cudaMemcpy(U[0][0], U_d, N*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // free device memory
    free_3d_cuda(U_d);
    free_3d_cuda(V_d);

    // end execution time measurement
    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start_run;
}

void jacobi_cuda_split(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, double *kt) {

    // device pointers for matiriies
    double *U_d[2];
    double *V_d[2];

    // start clock
    double start = omp_get_wtime();

    const int sizeBytes = (N/2)*N*N*sizeof(double);

    // allocate & populate device memory 
    for (int i=0; i<2; i++) {
        cudaSetDevice(i);
        // allocate

        // CHANGE TO DIRECT MALLOCS in CUDA
        CUDACHECK(cudaMalloc(U_d+i, sizeBytes));
        CUDACHECK(cudaMalloc(V_d+i, sizeBytes));

        //populate
        int start = i * (N/2);
        CUDACHECK(cudaMemcpyAsync(U_d[i], U[start][0], (N/2)*N*N*sizeof(double), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpyAsync(V_d[i], V[start][0], (N/2)*N*N*sizeof(double), cudaMemcpyHostToDevice));
    }

    // synchronize 
    for (int i=0; i<2; i++) {
        cudaSetDevice(i);
        CUDACHECK(cudaDeviceSynchronize());
    }

    // stop data-transfer clock after update
    *dt = omp_get_wtime()-start;

    // Enable peer-to-peer access and offload
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // define domain for GPU
    dim3 blocks(8, 8, 16); // a 8x8x8 block of threads
    dim3 grid((N) / blocks.x, (N) / blocks.y, (N/2) / blocks.z); // enough blocks to cover the whole 3D domain
    int m = 0;

    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    *kt = 0.0;
    double sk;
    while (m<K) {
        // call kernel
        sk = omp_get_wtime();
        for (int i=0; i<2; i++) {
            cudaSetDevice(i);
            poisson_split<<<grid, blocks>>>(U_d[i],V_d[i],U_d[1-i],div,del2,N,i);
        }
        // synchronize 
        for (int i=0; i<2; i++) {
            cudaSetDevice(i);
            CUDACHECK(cudaDeviceSynchronize());
        }

        // update total kernel runtime
        *kt += omp_get_wtime()-sk;

        // update (pointer-swap)
        for (int i = 0; i < 2; i++) {
            double *old = U_d[i];
            U_d[i] = V_d[i];
            V_d[i] = old;
        }
        m += 1;
        }
    
    // copy back to host and add to data transfer-time
    double out = omp_get_wtime();
    for (int i=0; i<2; i++) {
        cudaSetDevice(i);
        int start = (N/2)*(i);
        cudaMemcpyAsync(U[start][0], U_d[i], (N/2)*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    }

    // synchronize 
    for (int i=0; i<2; i++) {
        cudaSetDevice(i);
        CUDACHECK(cudaDeviceSynchronize());
    }

    // free device memory
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        free_3d_cuda(U_d[i]);
        free_3d_cuda(V_d[i]);
    }

    // update clocks
    *dt += omp_get_wtime()-out;
    *wt = omp_get_wtime()-start;

}

void jacobi_cuda_split_MPI(double ***U, 
                           double ***V, 
                           double ***F, 
                           int K, 
                           int N, 
                           double *wt, 
                           double *dt,
                           int argc, 
                           char *argv[],
                           char *output_filename,
                           char *output_prefix,
                           char *output_ext,
                           char *output_suffix,
                           int output_type) {

    double start = omp_get_wtime();

    // process info
    int myRank, nRanks, nDev = 0;
    int size = N*N;

    cudaGetDeviceCount(&nDev);

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //each process is using two GPUs
    double** sendbuff = (double**)malloc(nDev * sizeof(double*));
    double** recvbuff = (double**)malloc(nDev * sizeof(double*));
    ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nDev); 
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
   
    // initializing NCCL, group API is required around ncclCommInitRank as it is
    // called across multiple GPUs in each thread/process
    // nccl ranks use a global numbering
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
    }
    NCCLCHECK(ncclGroupEnd());


    // device pointers for matiriies
    double *U_d[2];
    double *V_d[2];

    const int sizeBytes = (N/4)*N*N*sizeof(double);
    // allocate & populate device memory 
    for (int i=0; i<nDev; i++) {
        // set device
        cudaSetDevice(i);

        // allocate
        CUDACHECK(cudaMalloc(U_d+i, sizeBytes));
        CUDACHECK(cudaMalloc(V_d+i, sizeBytes));

        //populate
        int start = ((2*myRank)+i)*(N/4);
        CUDACHECK(cudaMemcpy(U_d[i], U[start][0], (N/4)*N*N*sizeof(double), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(V_d[i], V[start][0], (N/4)*N*N*sizeof(double), cudaMemcpyHostToDevice));

        // create stream for each GPU
        CUDACHECK(cudaStreamCreate(s+i));
    }

    // populate buffers 
    if (myRank == 0) {
        cudaSetDevice(1);
        CUDACHECK(cudaMalloc(recvbuff+1, N*N*sizeof(double)));
    }
    else if (myRank == 1) {
        cudaSetDevice(0);
        CUDACHECK(cudaMalloc(recvbuff, N*N*sizeof(double)));
    }

    // stop data-transfer clock after update
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double end_t = omp_get_wtime();
    double duration_dt = end_t - start;
    *dt = duration_dt;

    // Enable peer-to-peer access and offload
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)

    // constants
    double del = 2.0/(double)(N);  // grid size 
    double div = (1.0/6.0);          // factor fo 1/6 from seven-point stencil formula
    double del2=(del*del);           // del squared 

    // kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(...);
    // define domain for internal points
    dim3 blocks(8, 4, 32); // a 8x8x8 block of threads
    dim3 grid((N) / blocks.x, (N) / blocks.y, ((N/4) / blocks.z)+1); 

    // for boundary kernel
    dim3 blocks2D(16, 16); // a 8x8x8 block of threads
    dim3 grid2D((N) / blocks2D.x, (N) / blocks2D.y); 

    double node_transfer;
    int m = 0;
    while (m<K) {
        // call kernel
        for (int i=0; i<2; i++) {
            cudaSetDevice(i);
            poisson_split_MPI<<<grid, blocks>>>(U_d[i],V_d[i],U_d[1-i],div,del2,N,i,myRank);
        }
        // synchronize 
        for (int i=0; i<2; i++) {
            cudaSetDevice(i);
            CUDACHECK(cudaDeviceSynchronize());
        }

        // exchange boundary data between nodes with NCCL

        NCCLCHECK(ncclGroupStart());
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        node_transfer = omp_get_wtime();
        if (myRank == 0) {
            NCCLCHECK(ncclSend(U_d[1]+((N/4-1)*N*N), size, ncclDouble, 2, comms[1], s[1]));
            NCCLCHECK(ncclRecv(recvbuff[1], size, ncclDouble, 2, comms[1], s[1]));
        }
        else if (myRank == 1) {
            NCCLCHECK(ncclSend(U_d[0], size, ncclDouble, 1, comms[0], s[0]));
            NCCLCHECK(ncclRecv(recvbuff[0], size, ncclDouble, 1, comms[0], s[0]));
        }
        NCCLCHECK(ncclGroupEnd());
        *dt += omp_get_wtime()-node_transfer;

        // synchronize
        for (int g = 0; g < nDev; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(s[g]);
        }

        // call boundary kernels 
        if (myRank == 0) {
            cudaSetDevice(1);
            poisson_boundary<<<grid2D, blocks2D>>>(U_d[1],V_d[1],recvbuff[1],div,del2,N,myRank);
        }
        else if (myRank == 1) {
            cudaSetDevice(0);
            poisson_boundary<<<grid2D, blocks2D>>>(U_d[0],V_d[0],recvbuff[0],div,del2,N,myRank);
        }
 
        // update (pointer-swap)
        for (int i = 0; i < 2; i++) {
            double *old = U_d[i];
            U_d[i] = V_d[i];
            V_d[i] = old;
        }
        m += 1;
        }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double dt2 = omp_get_wtime();
    int numElements = (N/2)*N*N;
    double *recvbuffMPI = (double *) malloc(2*numElements*sizeof(double)); // must have room for both processes in recv buffer

    // get all data onto one rank
    for (int i=0; i<2; i++) {
        cudaSetDevice(i);
        int start = ((2*myRank)+i)*(N/4);
        cudaMemcpy(U[start][0], U_d[i], (N/4)*N*N*sizeof(double), cudaMemcpyDeviceToHost);
    }

    // if elements exceed max, split in half and use two gather commands
    if (N < 704) {
        MPICHECK(MPI_Gather(U[N/2][0], numElements, MPI_DOUBLE, recvbuffMPI, numElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        // extract data from recvbuff
        if (myRank == 0) memcpy(U[N/2][0],recvbuffMPI+numElements,numElements*sizeof(double));
    }
    else if (N < 832) {
        int halfNumElements = numElements/2;
        MPICHECK(MPI_Gather(U[N/2][0], halfNumElements, MPI_DOUBLE, recvbuffMPI, halfNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPICHECK(MPI_Gather(U[3*N/4][0], halfNumElements, MPI_DOUBLE, recvbuffMPI + numElements, halfNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        // extract data from recvbuff
        if (myRank == 0) {
            memcpy(U[N/2][0],recvbuffMPI+halfNumElements,halfNumElements*sizeof(double));
            memcpy(U[3*N/4][0],recvbuffMPI+3*halfNumElements,halfNumElements*sizeof(double));
        }
    }
    else {
        int quartNumElements = numElements/4;
        MPICHECK(MPI_Gather(U[N/2][0], quartNumElements, MPI_DOUBLE, recvbuffMPI, quartNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPICHECK(MPI_Gather(U[5*N/8][0], quartNumElements, MPI_DOUBLE, recvbuffMPI + 2*quartNumElements, quartNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPICHECK(MPI_Gather(U[6*N/8][0], quartNumElements, MPI_DOUBLE, recvbuffMPI + 4*quartNumElements, quartNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPICHECK(MPI_Gather(U[7*N/8][0], quartNumElements, MPI_DOUBLE, recvbuffMPI + 6*quartNumElements, quartNumElements, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        // extract data from recvbuff
        if (myRank == 0) {
            memcpy(U[N/2][0],recvbuffMPI+quartNumElements,quartNumElements*sizeof(double));
            memcpy(U[5*N/8][0],recvbuffMPI+3*quartNumElements,quartNumElements*sizeof(double));
            memcpy(U[6*N/8][0],recvbuffMPI+5*quartNumElements,quartNumElements*sizeof(double));
            memcpy(U[7*N/8][0],recvbuffMPI+7*quartNumElements,quartNumElements*sizeof(double));
        }
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    *dt += omp_get_wtime()-dt2;


    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double dt_last = omp_get_wtime();

    // free device memory
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        free_3d_cuda(U_d[i]);
        free_3d_cuda(V_d[i]);
    }

    //freeing device memory
    for (int i=0; i<nDev; i++) {
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for (int i=0; i<nDev; i++) {
        ncclCommDestroy(comms[i]);
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    *dt += omp_get_wtime()-dt_last;
    *wt = omp_get_wtime()-start;

    //finalizing MPI
    MPICHECK(MPI_Finalize());


    // result
    if (output_type == 4) {
    if (myRank == 0) {
        output_ext = ".vtk";
        sprintf(output_filename, "%s_%d_%s%s", output_prefix, N,output_suffix, output_ext);
        fprintf(stderr, "Write VTK file to %s: ", output_filename);
        print_vtk(output_filename, N, U);
    }
    }
    else if (output_type == 1) {
    if (myRank == 0) {
        // print data 
        double memory = (3.0*8.0*N*N*N)/pow(10.0,6);
        double bandwidth = (1.0/(*wt))*((24.0*N*N*N)/pow(10.0,6));
        double bandwidth_ndt = (1.0/(*wt-*dt))*((24.0*N*N*N)/pow(10.0,6));
        printf("%d %f %f %f %f %f\n",N,*wt,*dt,memory,bandwidth,bandwidth_ndt); 
    }
    }
}