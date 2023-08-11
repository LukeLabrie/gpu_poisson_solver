#include <stdlib.h>
#include <omp.h>

double ***malloc_3d_device(int m, int n, int k, double **d) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double***) omp_target_alloc(m * sizeof(double **) + m * n * sizeof(double *),omp_get_default_device());

    
    if (p == NULL) {
        return NULL;
    }

    #pragma omp target is_device_ptr(p)
    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }

    double *a = (double*) omp_target_alloc(m * n * k * sizeof(double),omp_get_default_device());
    if (a == NULL) {
	    omp_target_free(p,omp_get_default_device());
	    return NULL;
    }

    #pragma omp target is_device_ptr(p,a)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    *d = a;
    return p;
}

void free_3d_device(double ***p) {
    int dev = omp_get_default_device();
    #pragma omp target is_device_ptr(p)
    {
    omp_target_free(p[0][0],dev);
    omp_target_free(p,dev);
    }
}
