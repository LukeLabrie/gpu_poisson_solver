#ifndef __ALLOC_3D_CUDA
#define __ALLOC_3D_CUDA

double *malloc_3d_cuda(int m, int n, int k);

#define HAS_FREE_3D_CUDA
void free_3d_cuda(double *array3D);

#endif /* __ALLOC_3D */
