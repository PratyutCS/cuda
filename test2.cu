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

__global__ void processData(unsigned char *d_data, size_t *d_sizes, int numElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index1 = idx*2;
    int index2 = (idx*2)+1;
    printf("index1 is: %d - index2 is: %d\n",index1, index2);
}

void print_binary(const unsigned char *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        for (int bit = 7; bit >= 0; bit--) {
            printf("%d", (data[i] >> bit) & 1);
        }
        printf(" ");
    }
    printf("\n");
}

int main() {
    int line_size;
    mpz_t * fileData = readFile("./input-4100.txt", &line_size);
    if (!fileData) {
        return 1;
    }

    unsigned char **binaryArray = (unsigned char **)malloc(line_size * sizeof(unsigned char *));
    size_t *binarySizes = (size_t *)malloc(line_size * sizeof(size_t));
    size_t totalSize = 0;
    
    for (int i = 0; i < line_size; i++) {
        binaryArray[i] = (unsigned char *)mpz_export(NULL, &binarySizes[i], 1, 1, 1, 0, fileData[i]);
        totalSize += binarySizes[i];
    }

    unsigned char *h_flatArray = (unsigned char *)malloc(totalSize);
    size_t offset = 0;
    for (int i = 0; i < line_size; i++) {
        memcpy(h_flatArray + offset, binaryArray[i], binarySizes[i]);
        offset += binarySizes[i];
    }

    // GPU memory allocation
    unsigned char *d_data;
    size_t *d_sizes;
    cudaMalloc(&d_data, totalSize);
    cudaMalloc(&d_sizes, line_size * sizeof(size_t));

    // Copy data to GPU
    cudaMemcpy(d_data, h_flatArray, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, binarySizes, line_size * sizeof(size_t), cudaMemcpyHostToDevice);

    // Launch kernel with optimized configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid = min((line_size + threadsPerBlock - 1) / threadsPerBlock, 2147483647);
    processData<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_sizes, line_size);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sizes);
    free(h_flatArray);
    for (int i = 0; i < line_size; i++) {
        free(binaryArray[i]);
        mpz_clear(fileData[i]);
    }
    free(binaryArray);
    free(binarySizes);
    free(fileData);

    return 0;
}

