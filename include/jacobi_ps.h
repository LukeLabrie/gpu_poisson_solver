/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_PS_H
#define _JACOBI_PS_H

void jacobi_ps(double ***U, 
               double ***V, 
               double ***F, 
               int k, 
               int N, 
               double *wt, 
               int *tot, 
               int threads
             );

#endif
