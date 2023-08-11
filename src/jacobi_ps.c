/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void jacobi_ps(double ***U, double ***V, double ***F, int K, int N, double *wt, int *tot, int threads) {

    // grid size
    double del = 2.0/(double)(N-1);
    double frac = 1.0/6.0;
    double del_square = del*del;

     // outer loop 
    int m, i, j, k;
    m = 0;
    double diff = INFINITY;
    omp_set_num_threads(threads);
    double start = omp_get_wtime();
    while ((m < K)) {
        
        //printf("it: %d \n",m);
        diff = 0.0;

        // inner loop
        #pragma omp parallel for default(none) \
                shared(U,V,F,N,del_square,frac) private(i,j,k) \
                reduction(+: diff) schedule(static)
        for (i = 1; i<N-1; i++) {
            for (j = 1; j<N-1; j++) {
                for (k = 1; k<N-1; k++) {
                    double old_U = U[i][j][k];
                    V[i][j][k] = (frac)*(U[i-1][j][k]+U[i+1][j][k]+U[i][j-1][k]+U[i][j+1][k]+U[i][j][k-1]+U[i][j][k+1]+(del_square)*F[i][j][k]);
                    diff += (V[i][j][k]-old_U)*(V[i][j][k]-old_U);
                }
            }
        }

        diff = sqrt(diff);
        m += 1;
        *tot += 1;

        // update U from V
        double ***old = U;
        U = V;
        V = old;
    }
    double end = omp_get_wtime();
    *wt = (end-start);
}


