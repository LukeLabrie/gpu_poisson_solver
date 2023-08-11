/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_OFFLOAD_H
#define _JACOBI_OFFLOAD_H

void jacobi_offload(double ***U, 
                    double ***V, 
                    double ***F, 
                    int k, 
                    int N, 
                    double *wt,
                    double *dt, 
                    int *tot
                    );

void jacobi_offload_dynamic(double ***U, 
                            double ***V, 
                            double ***F, 
                            int k, 
                            int N, 
                            double *wt, 
                            double *dt,
                            int *tot
                            );

void jacobi_offload_split(double ***U, 
                                  double ***V, 
                                  double ***F, 
                                  int k, 
                                  int N, 
                                  double *wt, 
                                  double *dt,
                                  double tol, 
                                  int *tot
                                );
                            
#endif
