#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>

// Reads hexadecimal numbers from a file (one per line) into an array of mpz_t numbers.
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
        // Remove newline if present.
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

// CUDA kernel: each block (with one thread) prints the numbers at positions index*2 and index*2+1.
__global__ void array(unsigned char *binaryArray, size_t totalSize, int *offsets, int *sizes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index1 = idx * 2;
    int index2 = idx * 2 + 1;

    // Print the number at index1, if it exists.
    if (index1 < count) {
        printf("Block %d - Number %d: ", idx, index1);
        for (int j = 0; j < sizes[index1]; j++) {
            printf("%02X ", binaryArray[offsets[index1] + j]);
        }
        printf("\n");
    }

    // Print the number at index2, if it exists.
    if (index2 < count) {
        printf("Block %d - Number %d: ", idx, index2);
        for (int j = 0; j < sizes[index2]; j++) {
            printf("%02X ", binaryArray[offsets[index2] + j]);
        }
        printf("\n");
    }
}

int main() {
    int line_size;
    mpz_t *fileData = readFile("./input-5.txt", &line_size);
    if (!fileData || line_size < 2) {
        printf("Error: Need at least two numbers in the input file.\n");
        return 1;
    }

    // Create arrays to store offsets and sizes for each exported mpz_t number.
    int *offsets = (int *) malloc(line_size * sizeof(int));
    int *sizes   = (int *) malloc(line_size * sizeof(int));
    size_t totalBytes = 0;
    unsigned char *host_flatArray = NULL;

    // Export each number's data to binary and record its offset and size.
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

    // Allocate device memory for the flat binary array, offsets, and sizes.
    unsigned char *d_flatArray;
    int *d_offsets, *d_sizes;
    cudaMalloc(&d_flatArray, totalBytes * sizeof(unsigned char));
    cudaMalloc(&d_offsets, line_size * sizeof(int));
    cudaMalloc(&d_sizes, line_size * sizeof(int));

    // Copy host data to device.
    cudaMemcpy(d_flatArray, host_flatArray, totalBytes * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, line_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, line_size * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of blocks (threads) to launch: one block for every two numbers.
    int numBlocks = line_size / 2;
    if (line_size % 2 != 0) {  // Add an extra block if there is an odd number of numbers.
        numBlocks++;
    }

    int threadsPerBlock = 1024;
    int blocksPerGrid = (line_size + threadsPerBlock - 1) / threadsPerBlock;
    array<<<blocksPerGrid, threadsPerBlock>>>(d_flatArray, totalBytes, d_offsets, d_sizes, line_size);
    cudaDeviceSynchronize();

    // Free allocated memory.
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

