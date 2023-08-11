#ifndef __ALLOC_3D_DEVICE
#define __ALLOC_3D_DEVICE

double ***malloc_3d_device(int m, int n, int k, double **d);

#define HAS_FREE_3D_DEVICE
void free_3d_device(double ***array3D);

#endif /* __ALLOC_3D */
