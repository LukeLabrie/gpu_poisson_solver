/* main.c - Poisson problem in 3D
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include "alloc3d.h"
#include "print.h" 
#include "initialize.h"
#include "jacobi_ps.h"
#include "jacobi_offload.h"
#include "jacobi_cuda.hpp"

#define N_DEFAULT 100



int main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;                      // grid size
    int 	iter_max = 1000;                    // sweeps
    double	tolerance;                          // tolerance (if relevant)
    double	start_T;                            // starting temp for inner points
    int		output_type = 0;                    // output format
    char *output_prefix = "poisson_res";     
    char *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    char* inf = "inf";
    int tot = 0;                                // max iterations
    int threads;                                // threads for cpu verision
    double t0,t1,kt;                               // run time & data transfer time 
    int method;                                    // solver method
    char *output_suffix;
    double ***U = NULL;
    double ***V = NULL;
    double ***F = NULL;
    double ***T = NULL;
    double *U_cont = NULL;
    double *V_cont = NULL;
    double *F_cont = NULL;

    /* get the paramters from the command line */
    N = atoi(argv[1]);	                    // grid size
    if (argv[2]==inf) iter_max = INFINITY;
    else iter_max  = atoi(argv[2]);         // max. no. of iterations
    start_T   = atof(argv[3]);              // start T for all inner grid points
	  output_type = atoi(argv[4]);            // ouput type
    method = atoi(argv[5]);                 // method 
    if (argc == 7) output_suffix = (argv[6]);              // file suffix
    if (argc == 8) threads = atoi(argv[7]);                // threads


    // allocate memory for i^th matrix
    if ( (U = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // allocate memory for (i+1)^th matrix
    if ( (V = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // allocate memory for (i+1)^th matrix
    if ( (F = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // debugger matrix
    if ( (T = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // populate matrices 
    set_U(U,N,start_T);
    set_U(V,N,start_T);
    set_F(F,N,start_T);

    // debugger
    // set_U(T,N,start_T);
    
    // solve
    if (method==1) jacobi_ps(U,V,F,iter_max,N,&t0,&tot,threads);
    else if (method==2) jacobi_offload_dynamic(U,V,F,iter_max,N,&t0,&t1,&tot);
    else if (method==3) jacobi_offload_split(U,V,F,iter_max,N,&t0,&t1,tolerance,&tot);
    else if (method==4) jacobi_cuda_single(U,V,F,iter_max,N,&t0,&t1,&kt);
    else if (method==5) jacobi_cuda_single_5(U,V,F,iter_max,N,&t0,&t1,&kt);
    else if (method==6) jacobi_cuda_split(U,V,F,iter_max,N,&t0,&t1,&kt);
    else if (method==7) jacobi_cuda_split_MPI(U,V,F,iter_max,N,&t0,&t1,argc,argv,output_filename,output_prefix,output_ext,output_suffix,output_type);

    bool mpi_method = (method == 7 || method == 8);

    // dump  results if wanted 
    switch(output_type) {
	  case 0:
	    // no output at all
	    break;
    case 1:
      if(!mpi_method) {
        // print data 
        double memory = (3.0*8.0*N*N*N)/pow(10.0,6);
        double bandwidth = (1.0/t0)*((24.0*N*N*N)/pow(10.0,6));
        double bandwidth_ndt = (1.0/(t0-t1))*((24.0*N*N*N)/pow(10.0,6));
        double bandwidth_k = (1.0/(kt))*((24.0*N*N*N)/pow(10.0,6));
        printf("%d %f %f %f %f %f %f %f\n",N,t0,t1,memory,bandwidth,bandwidth_ndt,kt,bandwidth_k); 
        break;
      }
	  case 3:
    if(!mpi_method) {
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s%s", output_prefix, N, output_suffix,output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, F);
	    break;
    }
	  case 4:
      if (!mpi_method) {
        output_ext = ".vtk";
        sprintf(output_filename, "%s_%d_%s%s", output_prefix, N,output_suffix, output_ext);
        fprintf(stderr, "Write VTK file to %s: ", output_filename);
        print_vtk(output_filename, N, U);
      }
	    break;
	  default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free_3d(U);
    free_3d(V);
    free_3d(F);
    
    return(0);
}
