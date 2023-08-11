/* 
    Funtions to initialize bounday(intiial) matricies
 */
#include <math.h>
#include <stdbool.h> 

void set_U(double ***U, int N, double T_0) {

    // inner loop
    int i, j, k;
    // set edges and radiator 
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {

                // variables 
                bool hot;

                // nonzero regions; hot edges and radiator 
                hot = ((j==N-1)||(k==N-1)||(k==0)||(i==N-1)||(i==0))&&(j!=0);
                // set temperatures 
                if (hot) U[i][j][k] = 20;
                else U[i][j][k] = T_0;

            }
        }
    }
}

void set_U_cont(double *U, int N, double T_0) {

    // inner loop
    int i, j, k;
    // set edges and radiator 
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {

                int global = i + (N*j) + k*(N*N);

                // variables 
                bool hot;

                // nonzero regions; hot edges and radiator 
                hot = ((j==N-1)||(k==N-1)||(k==0)||(i==N-1)||(i==0))&&(j!=0);
                // set temperatures 
                if (hot) U[global] = 20;
                else U[global] = T_0;

            }
        }
    }
}

void set_F(double ***F, int N, double T_0) {

    // set edges and radiator 
    int i, j, k;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {

                // variables 
                double x, y, z; 
                bool hot, rad;

                // location
                y = -1.0 + (2.0/(double)N)*(double)j;
                x = -1.0 + (2.0/(double)N)*(double)k;
                z = -1.0 + (2.0/(double)N)*(double)i;

                // nonzero regions; hot edges and radiator 
                hot = ((j==N-1)||(k==N-1)||(k==0)||(i==N-1)||(i==0))&&(j!=0);
                rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);

                // set temperatures 
                if (hot) F[i][j][k] = 20;
                else if (rad) F[i][j][k] = 200;
                else F[i][j][k] = T_0;

            }
        }
    }
}

void set_F_cont(double *F, int N, double T_0) {

    // set edges and radiator 
    int i, j, k;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {

                int global = i + (N*j) + k*(N*N);

                // variables 
                double x, y, z; 
                bool hot, rad;

                // location
                y = -1.0 + (2.0/(double)N)*(double)j;
                x = -1.0 + (2.0/(double)N)*(double)k;
                z = -1.0 + (2.0/(double)N)*(double)i;

                // nonzero regions; hot edges and radiator 
                hot = ((j==N-1)||(k==N-1)||(k==0)||(i==N-1)||(i==0))&&(j!=0);
                rad = (-1.0 <= x) && (x <= -3.0/8.0) && (-1.0 <= y) && (y <= -1.0/2.0) && (-2.0/3.0 <= z) && (z <= 0.0);

                // set temperatures 
                if (hot) F[global] = 20;
                else if (rad) F[global] = 200;
                else F[global] = T_0;

            }
        }
    }
}

