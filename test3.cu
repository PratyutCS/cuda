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

int main() {
    int line_size;
    mpz_t *fileData = readFile("./input-4100.txt", &line_size);
    if (!fileData || line_size < 2) {
        printf("Error: Need at least two numbers to multiply.\n");
        return 1;
    }

    // Convert first two numbers to binary arrays
    size_t size1, size2;
    unsigned char *num1 = (unsigned char *)mpz_export(NULL, &size1, 1, 1, 1, 0, fileData[3]);
    unsigned char *num2 = (unsigned char *)mpz_export(NULL, &size2, 1, 1, 1, 0, fileData[4]);

    size_t resultSize = size1 + size2;
    unsigned char *h_result = (unsigned char *)calloc(resultSize, sizeof(unsigned char));

    // Allocate GPU memory
    unsigned char *d_num1, *d_num2, *d_result;
    cudaMalloc(&d_num1, size1);
    cudaMalloc(&d_num2, size2);
    cudaMalloc(&d_result, resultSize);
    cudaMemset(d_result, 0, resultSize);

    // Copy data to GPU
    cudaMemcpy(d_num1, num1, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_num2, num2, size2, cudaMemcpyHostToDevice);

    // Launch kernel with a single thread
    multiplyLargeNumbers<<<1, 1>>>(d_num1, size1, d_num2, size2, d_result, resultSize);
    
    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Convert back to mpz_t and print result
    mpz_t result;
    mpz_init(result);
    mpz_import(result, resultSize, 1, 1, 1, 0, h_result);
    gmp_printf("Multiplication Result: %Zx\n", result);

        // Compute the expected result using GMP
    mpz_t expected_result;
    mpz_init(expected_result);
    mpz_mul(expected_result, fileData[3], fileData[4]);

    // Compare the GPU result with GMP result
    if (mpz_cmp(result, expected_result) == 0) {
        printf("GPU result matches GMP result! ✅\n");
    } else {
        printf("Mismatch in results! ❌\n");
        gmp_printf("Expected Result: %Zx\n", expected_result);
    }

    // Cleanup
    mpz_clear(expected_result);


    // Cleanup
    mpz_clear(result);
    cudaFree(d_num1);
    cudaFree(d_num2);
    cudaFree(d_result);
    free(h_result);
    free(num1);
    free(num2);
    for (int i = 0; i < line_size; i++) {
        mpz_clear(fileData[i]);
    }
    free(fileData);

    return 0;
}

