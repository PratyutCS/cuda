#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>

mpz_t * readFile(const char * fileName, int * size) {
    FILE * inputFile = fopen(fileName, "r");
    if (!inputFile) {
        perror("Error opening file");
        return NULL;
    }

    char * line = NULL;
    size_t line_length = 0;
    size_t line_count = 0;
    while (getline(&line, &line_length, inputFile) != -1) {
        line_count++;
    }

    mpz_t * array = (mpz_t *) malloc(line_count * sizeof(mpz_t));
    rewind(inputFile);

    for (size_t i = 0; i < line_count; i++) {
        mpz_init(array[i]);
        if (getline(&line, &line_length, inputFile) == -1) {
            perror("Error reading line");
            fclose(inputFile);
            free(line);
            for (size_t j = 0; j < i; j++) {
                mpz_clear(array[j]);
            }
            free(array);
            return NULL;
        }
        mpz_set_str(array[i], line, 16);
    }

    fclose(inputFile);
    free(line);
    *size = line_count;
    return array;
}

// Kernel to display binary data
__global__ void array(unsigned char *binaryArray, size_t totalSize) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU Binary Data:\n");
        for (size_t i = 0; i < totalSize; i++) {
            printf("%02X ", binaryArray[i]);
        }
        printf("\n");
    }
}

int main() {
    int line_size;
    mpz_t *fileData = readFile("./input-4.txt", &line_size);
    if (!fileData || line_size < 2) {
        printf("Error: Need at least two numbers to multiply.\n");
        return 1;
    }

    // Store sizes
    size_t totalBytes = 0;
    unsigned char *host_flatArray = NULL;

    for (int i = 0; i < line_size; i++) {
        size_t size;
        unsigned char *temp = (unsigned char *)mpz_export(NULL, &size, 1, 1, 1, 0, fileData[i]);
        totalBytes += size;
        host_flatArray = (unsigned char *)realloc(host_flatArray, totalBytes);
        memcpy(host_flatArray + (totalBytes - size), temp, size);
        free(temp);
    }

    // Allocate GPU memory
    unsigned char *d_flatArray;
    cudaMalloc(&d_flatArray, totalBytes * sizeof(unsigned char));

    // Copy data to GPU
    cudaMemcpy(d_flatArray, host_flatArray, totalBytes * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block, 1 thread
    array<<<1, 1>>>(d_flatArray, totalBytes);
    cudaDeviceSynchronize();

    // Free memory
    free(host_flatArray);
    cudaFree(d_flatArray);

    return 0;
}

