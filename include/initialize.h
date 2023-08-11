/* 
    Funtions to initialize bounday(intiial) matricies
 */

#ifndef _SET_U_H
#define _SET_U_H

void set_U(double ***U, int N, double start_T);
void set_U_cont(double *U, int N, double start_T);

#endif

#ifndef _SET_F_H
#define _SET_F_H

void set_F(double ***F, int N, double start_T);
void set_F_cont(double *F, int N, double start_T);

#endif