#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    float clockRate = prop.memoryClockRate; // in kiloHertz
    float busWidth = prop.memoryBusWidth;   // in bits

    // Convert clock rate to GigaHertz and bus width to bytes, then multiply by 2 for DDR memory
    float bandwidth = 2.0 * (clockRate * 1e3) * (busWidth / 8) / 1e9; // in GB/s

    std::cout << "Theoretical GPU memory bandwidth: " << bandwidth << " GB/s\n";

    return 0;
}
