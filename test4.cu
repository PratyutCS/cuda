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

void print_binary(const unsigned char *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        for (int bit = 7; bit >= 0; bit--) {
            printf("%d", (data[i] >> bit) & 1);
        }
        printf(" ");
    }
    printf("\n");
}

// Kernel function to multiply two large integers
__global__ void multiplyLargeNumbers(const unsigned char *d_num1, size_t size1,
                                     const unsigned char *d_num2, size_t size2,
                                     unsigned char *d_result, size_t resultSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) { // Run everything in a single thread
        for (int i = size1 - 1; i >= 0; i--) {
            int carry = 0;
            for (int j = size2 - 1; j >= 0; j--) {
                size_t pos = i + j + 1;
                int product = d_num1[i] * d_num2[j] + d_result[pos] + carry;
                d_result[pos] = product & 0xFF; // Store lower byte
                carry = product >> 8;          // Extract carry
            }
            // Add remaining carry
            d_result[i] += carry;
        }
    }
}

__global__ void array( unsigned char ** binaryArray, size_t *binarySize){

}

int main() {
    int line_size;
    mpz_t *fileData = readFile("./input-4100.txt", &line_size);
    if (!fileData || line_size < 2) {
        printf("Error: Need at least two numbers to multiply.\n");
        return 1;
    }

    size_t *binSize = (size_t *)malloc(line_size * sizeof(size_t));

    unsigned char **binArr = (unsigned char **)malloc(line_size * sizeof(unsigned char *));

    for(int i=0 ; i<line_size ; i++){
        binArr[i] = (unsigned char *)mpz_export(NULL, &binSize[i], 1, 1, 1, 0, fileData[i]);
    }

    unsigned char **d_binArr;
    size_t *d_binSize;
    cudaMalloc(&d_binArr, line_size * sizeof(unsigned char *));
    cudaMalloc(&d_binSize, line_size * sizeof(size_t));

    cudaMemcpy(d_binArr, binArr, line_size * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binSize, binSize, line_size * sizeof(size_t), cudaMemcpyHostToDevice);

    array<<<1,1<>>>(d_binArr, d_binSize);
    
    cudaDeviceSynchronize();

    return 0;
}

