/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "alloc3d_device.h"
#include <cuda.h>
#include <stdbool.h>

void jacobi_offload(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, int *tot) {

    // grid size
    double del = 2.0/(double)(N-1);

     // outer loop
    int m;
    m = 0;
    double dev = (1.0/6.0);
    double del2=(del*del);

    double start = omp_get_wtime();

    #pragma omp target enter data map(to:F[0:N][0:N][0:N],U[0:N][0:N][0:N],V[0:N][0:N][0:N]) \
                                  map(alloc: U[0:N][0:N][0:N],V[0:N][0:N][0:N])
    *dt = omp_get_wtime() - start;  
    while (m < K) {
            #pragma omp target teams loop bind(teams) 
                for (int j = 1; j<N-1; j++) {
                    #pragma omp loop bind(parallel)//collapse(2)
                    for (int i = 1; i<N-1; i++) {
                        for (int k = 1; k<N-1; k++){
                        V[i][j][k] = dev*(U[i-1][j][k]+U[i+1][j][k]+U[i][j-1][k]+U[i][j+1][k]+U[i][j][k-1]+U[i][j][k+1]+del2*F[i][j][k]);
                    }
                }
            }

        // update iterations
        m += 1;

        // update U from V
        double ***old = U;
        U = V;
        V = old;
     }
     double start2 = omp_get_wtime();
     #pragma omp target exit data map(release:F[0:N][0:N][0:N],U[0:N][0:N][0:N],V[0:N][0:N][0:N]) \
                                  map(from: U[0:N][0:N][0:N],V[0:N][0:N][0:N])
     *dt += omp_get_wtime()-start2;
     double end = omp_get_wtime();
     *wt = end-start;
     *tot = m;

    }


void jacobi_offload_dynamic(double ***U, double ***V, double ***F, int K, int N, double *wt, double *dt, int *tot) {

    // device pointers for matiriies
    double ***U_d;
    double ***V_d;
    double ***F_d;
    double *u_d;
    double *v_d;
    double *f_d;

    double start = omp_get_wtime();
    // allocate on device
    U_d = malloc_3d_device(N, N, N, &u_d); // Allocate A_d on device
    V_d = malloc_3d_device(N, N, N, &v_d); // Allocate A_d on device
    F_d = malloc_3d_device(N, N, N, &f_d); // Allocate A_d on device

    // initialize on device
    omp_target_memcpy(u_d, U[0][0], N * N * N * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(v_d, V[0][0], N * N * N * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(f_d, F[0][0], N * N * N * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());

    double end_t = omp_get_wtime();
    double duration_dt = end_t - start;
    *dt = duration_dt;

    // grid size
    double del = 2.0/(double)(N-1);

     // outer loop
    int m;
    m = 0;
    double dev = (1.0/6.0);
    double del2=(del*del);
    

    while (m < K) {
            #pragma omp target teams loop is_device_ptr(U_d,V_d,F_d)
                for (int j = 1; j<N-1; j++) {
                    #pragma omp loop bind(parallel)
                    for (int i = 1; i<N-1; i++) {
                        for (int k = 1; k<N-1; k++){
                        V_d[i][j][k] = dev*(U_d[i-1][j][k]+U_d[i+1][j][k]+U_d[i][j-1][k]+U_d[i][j+1][k]+U_d[i][j][k-1]+U_d[i][j][k+1]+del2*F_d[i][j][k]);
                    }
                }
            }

        // update iterations
        m += 1;

        // update U from V
        

        double ***old = U_d;
        U_d = V_d;
        V_d = old;
    

     }
     *tot = m;

     omp_target_memcpy(U[0][0],u_d, N * N * N * sizeof(double),0, 0, omp_get_initial_device(),omp_get_default_device());

    double end = omp_get_wtime();
    double duration = end-start;
    *wt = duration;

    free_3d_device(U_d);
    free_3d_device(V_d);
    free_3d_device(F_d);
}


void jacobi_offload_split(double ***U, double ***V, double ***F, int K, int N_tot, double *wt, double *dt, double tol, int *tot) {


    double t0 = omp_get_wtime();

    // first half
    omp_set_default_device(0);
    int N = N_tot/2;
    
    // device pointers for matiriies
    double ***U_d1;
    double ***V_d1;
    double ***F_d1;
    double *u_d1;
    double *v_d1;
    double *f_d1;
    // allocate on device

    U_d1 = malloc_3d_device(N, N_tot, N_tot, &u_d1); // Allocate A_d on device
    V_d1 = malloc_3d_device(N, N_tot, N_tot, &v_d1); // Allocate A_d on device
    F_d1 = malloc_3d_device(N, N_tot, N_tot, &f_d1); // Allocate A_d on device

    // initialize on device
    omp_target_memcpy(u_d1, U[0][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(v_d1, V[0][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(f_d1, F[0][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());

    // second half
    omp_set_default_device(1);
     //device pointers for matiriies
    double ***U_d2;
    double ***V_d2;
    double ***F_d2;
    double *u_d2;
    double *v_d2;
    double *f_d2;

    // allocate on device
    U_d2 = malloc_3d_device(N, N_tot, N_tot, &u_d2); // Allocate A_d on device
    V_d2 = malloc_3d_device(N, N_tot, N_tot, &v_d2); // Allocate A_d on device
    F_d2 = malloc_3d_device(N, N_tot, N_tot, &f_d2); // Allocate A_d on device

    // initialize on device
    omp_target_memcpy(u_d2, U[N][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(v_d2, V[N][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(f_d2, F[N][0], N * N_tot * N_tot * sizeof(double),0, 0, omp_get_default_device(), omp_get_initial_device());

    // update data transfer time 
    *dt = omp_get_wtime() - t0;

    // grid size
    double del = 2.0/(double)(N_tot-1);
     // outer loop
    int m;
    m = 0;
    double dev = (1.0/6.0);
    double del2=(del*del);

    // Enable peer-to-peer access and offload
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)

    
    while (m < K) {
            
            omp_set_default_device(0);
            #pragma omp target teams loop is_device_ptr(U_d1,V_d1,F_d1) \
                     nowait
            for (int i = 1; i<N-1; i++) {
            #pragma omp loop bind(parallel)
            for (int j = 1; j<N_tot-1; j++) {
            for (int k = 1; k<N_tot-1; k++){
            V_d1[i][j][k] =  dev*(U_d1[i-1][j][k]+U_d1[i+1][j][k]+U_d1[i][j-1][k]+
                                    U_d1[i][j+1][k]+U_d1[i][j][k-1]+U_d1[i][j][k+1]+del2*F_d1[i][j][k]);

            }}}

            #pragma omp target teams loop is_device_ptr(U_d1,V_d1,F_d1,U_d2,V_d2) \
                     nowait
            for (int i = N-1; i<N; i++) {
            #pragma omp loop bind(parallel)
            for (int j = 1; j<N_tot-1; j++) {
            for (int k = 1; k<N_tot-1; k++){
            V_d1[i][j][k] =  dev*(U_d1[i-1][j][k]+U_d2[0][j][k]+U_d1[i][j-1][k]+
                                   U_d1[i][j+1][k]+U_d1[i][j][k-1]+U_d1[i][j][k+1]+del2*F_d1[i][j][k]);
            }}}

            omp_set_default_device(1);
            #pragma omp target teams loop is_device_ptr(U_d2,V_d2,F_d2,U_d1) \
                     nowait
            for (int i = 0; i<1; i++) {
            #pragma omp loop bind(parallel)
            for (int j = 1; j<N_tot-1; j++) {
            for (int k = 1; k<N_tot-1; k++){
                V_d2[i][j][k] = dev*(U_d1[N-1][j][k]+U_d2[i+1][j][k]+U_d2[i][j-1][k]+U_d2[i][j+1][k]+U_d2[i][j][k-1]+U_d2[i][j][k+1]+del2*F_d2[i][j][k]);
            }}}

            
            #pragma omp target teams loop is_device_ptr(U_d2,V_d2,F_d2) \
                     nowait
            for (int i = 1; i<N-1; i++) {
            #pragma omp loop bind(parallel)
            for (int j = 1; j<N_tot-1; j++) {
            for (int k = 1; k<N_tot-1; k++){
                V_d2[i][j][k] = dev*(U_d2[i-1][j][k]+U_d2[i+1][j][k]+U_d2[i][j-1][k]+U_d2[i][j+1][k]+U_d2[i][j][k-1]+U_d2[i][j][k+1]+del2*F_d2[i][j][k]);
            }}}
            
            #pragma omp taskwait
            {
            m += 1;
            double ***old1 = U_d1;
            U_d1 = V_d1;
            V_d1 = old1;
            double ***old2 = U_d2;
            U_d2 = V_d2;
            V_d2 = old2;
            }
        
     }

    double end_run = omp_get_wtime();
    *tot = m;
    omp_target_memcpy(U[0][0],u_d1, N * N_tot * N_tot  * sizeof(double),0,0,omp_get_initial_device(),0);
    omp_target_memcpy(U[N][0],u_d2, N * N_tot * N_tot  * sizeof(double),0,0,omp_get_initial_device(),1);
    

    omp_set_default_device(0);
    free_3d_device(U_d1);
    free_3d_device(V_d1);
    free_3d_device(F_d1);
    
    omp_set_default_device(1);
    free_3d_device(U_d2);
    free_3d_device(V_d2);
    free_3d_device(F_d2);

    // wall time
    *dt += omp_get_wtime() - end_run;
    *wt = omp_get_wtime() - t0;

}

void jacobi_offload_norm(double ***U, double ***V, double ***F, int iter_max, int N, double *start_T, double tol, int *tot){
    // grid size
    double del = 2.0/(double)N;

     // outer loop 
   
   
    int n = N-1;
    
    
    double dev = (1.0/6.0);
    double del2=(del*del);
    int m;
    m = 0;
    
    double start;
    double end;
    start = omp_get_wtime(); 
    //
    #pragma omp target data map(to: F[0:N][0:N][0:N]) map(tofrom: U[0:N][0:N][0:N], V[0:N][0:N][0:N]) //map(tofrom:norm) 
    {
    double old_U;
    double norm = 99999999999.0;
    //start2 = omp_get_wtime(); 
    while ((m < iter_max) && (norm > tol)) {
        
        //printf("it: %d \n",m);
        double sum = 0.0;
        //printf("norm:%f, iter:%d \n", norm, m);
        //printf("%f",diff);
        //#pragma omp parallel for schedule(static)\
        private(i,j,k,old_U) reduction(+ : diff)
        //#pragma omp enter target data map(to: U[0:N][0:N][0:N], F[0:N][0:N][0:N]) map(alloc: V[0:N][0:N][0:N])
            //#pragma omp target teams loop \
            map( F[0:N][0:N][0:N]) map(tofrom: U[0:N][0:N][0:N], V[0:N][0:N][0:N])
            #pragma omp target teams loop collapse(2) reduction(+ : sum) //map(tofrom:norm) map(to: F[0:N][0:N][0:N]) map(tofrom: U[0:N][0:N][0:N], V[0:N][0:N][0:N]) map(tofrom:start2) map(tofrom:end2)
            //num_teams(6000) thread_limit(128)    
            
                
                for (int i = 1; i<n; i++) {
                    
                    for (int j = 1; j<n; j++) { 
                
                        #pragma omp loop bind(parallel) 
                        for (int k = 1; k<n; k++){
                        
                        old_U = U[i][j][k];
                        V[i][j][k] = dev*(U[i-1][j][k]+U[i+1][j][k]+U[i][j-1][k]+U[i][j+1][k]+U[i][j][k-1]+U[i][j][k+1]+del2*F[i][j][k]);
                        //#pragma omp critical
                        double diff = V[i][j][k]-old_U;

                        sum += diff*diff;
                       
                        //printf("i = %d, j= %d, k=%d, diff=%4f, threadId = %d \n", i, j, k, diff, omp_get_thread_num());
                    }
                }

            }//end of parallel for
        //#pragma omp target exit data map(release: F[0:N][0:N][0:N]) map(from:V[0:N][0:N][0:N], U[0:N][0:N][0:N])
        
        
        //printf("%d \n",diff);
        m += 1;
        norm = sqrt(sum);

        // update U from V
        double ***old = U;
        U = V;
        V = old;
        //printf("norm:%f, sum:%f, iter:%d \n", norm,sum, m);
    }//end while
    //end2 = omp_get_wtime(); 
    }//end of data block
    end = omp_get_wtime(); 
    printf("GPU Problem size: %d, iter: %d, time (s): %f,  #of teams: %d, #of threads/team: %d \n", N, iter_max, end-start ,omp_get_team_size, omp_get_num_threads);
    
}

