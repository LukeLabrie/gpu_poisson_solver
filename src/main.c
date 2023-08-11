/* main.c - Poisson problem in 3D
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h" 
#include "initialize.h"
#include "jacobi_ps.h"
#include "jacobi_offload.h"

#define N_DEFAULT 100

int main(int argc, char *argv[]) {


    int 	N = N_DEFAULT;                      // grid size
    int 	iter_max = 1000;                    // sweeps
    double	tolerance;                          // tolerance (if relevant)
    double	start_T;                            // starting temp for inner points
    int		output_type = 0;                    // output format
    char	*output_prefix = "poisson_res";     
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***U = NULL;                        // matricies
    double 	***V = NULL;
    double 	***F = NULL;
    char* inf = "inf";
    int tot = 0;                                // max iterations
    int threads;                                // threads for cpu verision
    double t0,t1;                               // run time & data transfer time 
    int method;                                 // solver method

    /* get the paramters from the command line */
    N = atoi(argv[1]);	                    // grid size
    if (argv[2]==inf) iter_max = INFINITY;
    else iter_max  = atoi(argv[2]);         // max. no. of iterations
    tolerance = atof(argv[3]);              // tolerance
    start_T   = atof(argv[4]);              // start T for all inner grid points
	output_type = atoi(argv[5]);            // ouput type
    method = atoi(argv[6]);
    if (argc == 8) {
	threads = atoi(argv[7]);                // threads
    }

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

    // populate boundary matrix 
    set_U(U,N,start_T);
    set_U(V,N,start_T);
    set_F(F,N,start_T);
    

    // solve
    if (method==1) jacobi_ps(U,V,F,iter_max,N,&t0,tolerance,&tot,threads);
    else if (method==2) jacobi_offload_dynamic(U,V,F,iter_max,N,&t0,&t1,tolerance,&tot);
    else if (method==3) jacobi_offload_split(U,V,F,iter_max,N,&t0,tolerance,&tot);

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
    //case 1:
    //    // convergence tests, just prints number of iterations to tolerance
    //    if (ALGO==3||ALGO==4||ALGO==5||ALGO==8||ALGO==6) printf("%d %d %f %d %f\n",N,tot,t0,t1, threads); 
    //    else printf("%d %d %f %f\n",N,tot,t0,t1); 
    //    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, U);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, U);
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
