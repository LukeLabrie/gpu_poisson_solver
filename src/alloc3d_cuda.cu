#include <stdlib.h>
#include <cuda.h>

double *malloc_3d_cuda(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    // Calculate the total number of elements in the 3D matrix
    const int numElements = m * n * k;

    // Calculate the size of memory required in bytes
    const int sizeBytes = numElements * sizeof(double);

    // Allocate memory on the device (GPU)
    double *d_matrix;
    cudaMalloc((void**)&d_matrix, sizeBytes);

    // Check if the memory allocation was successful
    if (d_matrix == nullptr) {
        return NULL;
    }

    return d_matrix;

}

void free_3d_cuda(double *array3D) {
    cudaFree(array3D);
}

