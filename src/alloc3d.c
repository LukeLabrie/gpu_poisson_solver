#include <stdlib.h>

double ***malloc_3d(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double***) malloc(m * sizeof(double **) +
                                     m * n * sizeof(double *));
    if (p == NULL) {
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }

    double *a = (double*) malloc(m * n * k * sizeof(double));
    if (a == NULL) {
	free(p);
	return NULL;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    return p;
}

double *malloc_3d_cont(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    // Calculate the total number of elements in the 3D matrix
    const int numElements = m * n * k;

    // Calculate the size of memory required in bytes
    const int sizeBytes = numElements * sizeof(double);

    // Allocate memory on the device (GPU)
    double *d_matrix = (double *) malloc(sizeBytes);

    // Check if the memory allocation was successful
    if (d_matrix == nullptr) {
        return NULL;
    }

    return d_matrix;
}

void
free_3d(double ***p) {
    free(p[0][0]);
    free(p);
}


