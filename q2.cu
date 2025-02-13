#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of available devices

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev); // Get properties of the device

        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Max grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("Max block dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);

        // Calculate total maximum threads
        int maxThreads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
        printf("Total maximum threads: %d\n\n", maxThreads);
    }

    return 0;
}
