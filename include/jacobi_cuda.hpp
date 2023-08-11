/* jacobi_cuda.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_CUDA_H
#define _JACOBI_CUDA_H

int idx3D(int i, int j, int k, int n);

void jacobi_cuda_single(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_single_2(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_single_3(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_single_4(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_single_5(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_split(double ***U, double ***V, double ***F, int k, int N, double *wt, double *dt, double *kt);

void jacobi_cuda_split_MPI(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, int argc, char *argv[], char *output_filename, char *output_prefix, char *output_ext, char *output_suffix, int output_type);       

__global__
void poisson_single(double *u_d, double *v_d, double *f_d, double div, double del2, int n);

__global__
void poisson_single_2(double *u_d, double *v_d, double *f_d, double div, double del2, int n);

__global__
void poisson_single_3(double *u_down, double *u, double *u_up, double *v, double *f, double div, double del2, int n);

__global__
void poisson_single_4(double *u_d, double *v_d, double *f_d, double div, double del2, int n);

__global__
void poisson_single_5(double *u_d, double *v_d, double div, double del2, int n);

__global__
void poisson_split(double *u_d1, 
                   double *v_d, 
                   double *u_d2, 
                   double div, 
                   double del2, 
                   int n, 
                   int dev);

__global__
void poisson_boundary(double *u_d, 
                      double *v_d, 
                      double *u_d2,
                      double div,   
                      double del2,  
                      int n,        
                      int rank    
                        );

__global__
void poisson_split_MPI(double *u_d1, // matricies for 1st GPU
                       double *v_d, 
                       double *u_d2, // matricies for 2nd GPU
                       double div,   // coefficients for stencil formula
                       double del2,  
                       int n,        // dimensions of XY region   
                       int dev,      // device number
                       int rank      // MPI rank 
                   );
                            
#endif
