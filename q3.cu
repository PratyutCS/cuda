#include <stdio.h>
#include <cuda.h>

__global__ void timeKernel(long long *d_startTimes) {
    // Get start time in clock cycles
    long long start = clock64();

    // Prevent optimization by adding a dummy operation
    long long lol;
    for (long long i = 0; i < 100000000000000000; i++) {
        lol = i / 100000000000000000;
    }
    lol = lol/2;

    // Store start time for each thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_startTimes[idx] = start;
}

int main() {
    const int numThreadsPerBlock = 1024;
    const int numBlocks = 500;
    const int totalThreads = numThreadsPerBlock * numBlocks;

    long long *d_startTimes, h_startTimes[totalThreads];

    // Allocate memory on GPU
    cudaMalloc(&d_startTimes, totalThreads * sizeof(long long));

    // Launch kernel with multiple threads
    timeKernel<<<numBlocks, numThreadsPerBlock>>>(d_startTimes);
    cudaDeviceSynchronize();

    // Copy start times back to CPU
    cudaMemcpy(h_startTimes, d_startTimes, totalThreads * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_startTimes);

    // Get device properties to retrieve clock rate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clockRate = prop.clockRate * 1.0e3; // Convert kHz to Hz

    // Convert clock cycles to nanoseconds and print results
    printf("Thread Start Times (subset shown):\n");
    for (int i = 0; i < numThreadsPerBlock * numBlocks; i++) { // Print only first 10 threads to reduce output
        double startTimeNano = (h_startTimes[i] / clockRate) * 1.0e9;
        printf("Thread %d: %lf nanoseconds\n", i, startTimeNano);
    }

    return 0;
}

