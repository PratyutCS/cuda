#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>

// Reads hex numbers (one per line) from a file into an array of mpz_t numbers.
mpz_t * readFile(const char * fileName, int * size) {
    FILE * inputFile = fopen(fileName, "r");
    if (!inputFile) {
        perror("Error opening file");
        return NULL;
    }

    char * line = NULL;
    size_t line_length = 0;
    size_t line_count = 0;
    // Count the number of lines
    while (getline(&line, &line_length, inputFile) != -1) {
        line_count++;
    }
    free(line);
    
    mpz_t * array = (mpz_t *) malloc(line_count * sizeof(mpz_t));
    rewind(inputFile);
    
    line = NULL;
    line_length = 0;
    // Read each line and convert from hex string to mpz_t
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
        // Remove newline if present
        size_t len = strlen(line);
        if(len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        mpz_set_str(array[i], line, 16);
    }
    
    fclose(inputFile);
    free(line);
    *size = line_count;
    return array;
}

// Kernel that prints the entire binary array and then prints each number (using offsets and sizes) on a separate line.
__global__ void array(unsigned char *binaryArray, size_t totalSize, int *offsets, int *sizes, int count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU Binary Data:\n");
        for (size_t i = 0; i < totalSize; i++) {
            printf("%02X ", binaryArray[i]);
        }
        printf("\n\n");
        
        // Print each individual number in hex using its offset and size.
        for (int i = 0; i < count; i++) {
            printf("Number %d: ", i);
            int off = offsets[i];
            int sz = sizes[i];
            for (int j = 0; j < sz; j++) {
                printf("%02X ", binaryArray[off + j]);
            }
            printf("\n");
        }
    }
}

int main() {
    int line_size;
    mpz_t *fileData = readFile("./input-4.txt", &line_size);
    if (!fileData || line_size < 2) {
        printf("Error: Need at least two numbers to multiply.\n");
        return 1;
    }

    // Create arrays to hold the offset and size of each number in the flat binary array.
    int *offsets = (int *) malloc(line_size * sizeof(int));
    int *sizes   = (int *) malloc(line_size * sizeof(int));
    size_t totalBytes = 0;
    unsigned char *host_flatArray = NULL;

    // For each mpz_t, export its data to a temporary buffer, record its size and offset,
    // and copy the data into a growing flat array.
    for (int i = 0; i < line_size; i++) {
        size_t sz;
        unsigned char *temp = (unsigned char *) mpz_export(NULL, &sz, 1, 1, 1, 0, fileData[i]);
        offsets[i] = totalBytes;
        sizes[i] = (int) sz;
        totalBytes += sz;
        host_flatArray = (unsigned char *) realloc(host_flatArray, totalBytes);
        memcpy(host_flatArray + offsets[i], temp, sz);
        free(temp);
    }

    // Allocate device memory for the flat array, offsets, and sizes.
    unsigned char *d_flatArray;
    int *d_offsets, *d_sizes;
    cudaMalloc(&d_flatArray, totalBytes * sizeof(unsigned char));
    cudaMalloc(&d_offsets, line_size * sizeof(int));
    cudaMalloc(&d_sizes, line_size * sizeof(int));

    // Copy host arrays to device memory.
    cudaMemcpy(d_flatArray, host_flatArray, totalBytes * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, line_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, line_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 1 thread.
    array<<<1, 1>>>(d_flatArray, totalBytes, d_offsets, d_sizes, line_size);
    cudaDeviceSynchronize();

    // Free host and device memory.
    free(host_flatArray);
    free(offsets);
    free(sizes);
    for (int i = 0; i < line_size; i++) {
        mpz_clear(fileData[i]);
    }
    free(fileData);
    cudaFree(d_flatArray);
    cudaFree(d_offsets);
    cudaFree(d_sizes);

    return 0;
}

