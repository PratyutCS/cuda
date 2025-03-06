#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <gmp.h>
#include <stdbool.h>
#include <ctype.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel that directly implements schoolbook multiplication
__global__ void kernel(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, uint32_t* d_result) {
    // Clear the result array first
    for (size_t i = 0; i < s1 + s2; i++) {
        d_result[i] = 0;
    }
    
    // Standard long multiplication algorithm
    for (size_t i = 0; i < s1; i++) {
        uint64_t carry = 0;
        
        for (size_t j = 0; j < s2; j++) {
            // Calculate position in result array
            size_t pos = (s1 + s2) - 1 - (i + j);
            
            // Calculate the product and add to existing value with carry
            uint64_t prod = (uint64_t)num1[s1-1-i] * (uint64_t)num2[s2-1-j] + 
                           d_result[pos] + carry;
            
            // Store the lower 32 bits
            d_result[pos] = (uint32_t)(prod & 0xFFFFFFFF);
            
            // Carry the upper bits
            carry = prod >> 32;
        }
        
        // Handle remaining carry
        size_t idx = (s1+s2) - 1 - (i + s2);
        while (carry > 0 && idx < (s1+s2)) {
            uint64_t sum = (uint64_t)d_result[idx] + carry;
            d_result[idx] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
            if (idx == 0) break;
            idx--;
        }
    }
    
    // Note: We don't handle leading zeros here as we return the full array
    // Leading zeros are handled on the host side if needed
}

// String generation function (unchanged)
char* gen(long long length) {
    char* hex_string = (char*)malloc((length + 1) * sizeof(char));
    if (hex_string == NULL) {
        return NULL;
    }
    
    const char hex_chars[] = "123456789ABCDEF";
    for (long long i = 0; i < length; i++) {
        long long random_index = rand() % 15;
        hex_string[i] = hex_chars[random_index];
    }
    hex_string[length] = '\0';
    return hex_string;
}

// Array to hex conversion (unchanged)
char* uint32_array_to_hex(uint32_t* arr, size_t size) {
    if (!arr || size == 0) return NULL;
    
    size_t hex_str_size = (size * 8) + 1;
    char* hex_str = (char*)malloc(hex_str_size);
    if (!hex_str) {
        perror("Memory allocation failed");
        return NULL;
    }
    
    char* ptr = hex_str;
    for (size_t i = 0; i < size; i++) {
        sprintf(ptr, "%08X", arr[i]);
        ptr += 8;
    }
    *ptr = '\0';
    
    return hex_str;
}

// String comparison function (unchanged)
bool compare_strings(char *str1, char *str2) {
    size_t len2 = strlen(str2);
    char *str3 = (char*)malloc(len2 + 1 * sizeof(char));
    if (!str3) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < len2; i++) {
        str3[i] = toupper((unsigned char)str2[i]);
    }
    str3[len2] = '\0';

    size_t length1 = strlen(str1);
    size_t length2_actual = strlen(str3);
    
    size_t min_length = (length1 < length2_actual) ? length1 : length2_actual;

    bool flag = true;
    
    for (size_t i = 0; i < min_length; i++) {
        if (str1[length1 - 1 - i] != str3[length2_actual - 1 - i]) {
            printf("Difference found: %c from str1 vs %c from str3\n",
                   str1[length1 - 1 - i], str3[length2_actual - 1 - i]);
            FILE *file = fopen("output.txt", "a");
            if (file) {
                fprintf(file, "Difference found: %c from str1 vs %c from str3\n",
                    str1[length1 - 1 - i], str3[length2_actual - 1 - i]);
                fclose(file);
            }
            flag = false;
        }
    }
    free(str3);
    return flag;
}

mpz_t * readFile(const char * fileName, int * size) {
    //printf("[READFILE] inputfileName is: %s\n", fileName);
    FILE * inputFile = fopen(fileName, "r");
  
    if (!inputFile) {
      perror("Error opening file");
      return NULL;
    }
  
    char * line = NULL;
    size_t line_length = 0;
    size_t line_count = 0;
  
    while (getline( & line, & line_length, inputFile) != -1) {
      line_count++;
    }
  
    //printf("[READFILE] line count is: %zd\n", line_count);
    mpz_t * array = (mpz_t * ) malloc(line_count * sizeof(mpz_t));
    rewind(inputFile);
  
    for (size_t i = 0; i < line_count; i++) {
      mpz_init(array[i]);
      if (getline( & line, & line_length, inputFile) == -1) {
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
    * size = line_count;
    return array;
}


int main() {
    int line_size;
    mpz_t* fileData = readFile("./input-1000.txt", &line_size);

    long row = 0;
    for (int i = line_size; i >= 1; i = (i + 1) / 2) {
        row++;
        if (i == 1)
            break;
    }

    uint32_t*** aoa = (uint32_t***) malloc(row * sizeof(uint32_t**));
    size_t** s = (size_t**) malloc(row * sizeof(size_t*));
    aoa[0] = (uint32_t**) malloc(line_size * sizeof(uint32_t*));
    s[0] = (size_t*) malloc(line_size * sizeof(size_t));

    // Get size of GPU memory
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory: %zu MB free out of %zu MB total\n", 
           free_mem/(1024*1024), total_mem/(1024*1024));
    
    // Leave some memory for other operations
    size_t max_usable_mem = free_mem * 0.9;  // Use only 90% of available memory
    
    for (int i = 0; i < line_size; i++) {
        size_t s1 = ((mpz_sizeinbase(fileData[i], 2) + 31) / 32);
        s[0][i] = s1;
        aoa[0][i] = (uint32_t*) malloc(s1 * sizeof(uint32_t));
        mpz_t temp;
        mpz_init_set(temp, fileData[i]);
        for (size_t j = 0; j < s1; j++) {
            aoa[0][i][j] = (uint32_t) mpz_get_ui(temp);
            mpz_fdiv_q_2exp(temp, temp, 32);
        }
        mpz_clear(temp);
    }

    int prev = line_size;
    int n_line_size = (line_size + 1) / 2;

    for(int i=1; i<row; i++){
        printf("Running level %d - %d\n", i, n_line_size);
        aoa[i] = (uint32_t**) malloc(n_line_size * sizeof(uint32_t*));
        s[i] = (size_t*) malloc(n_line_size * sizeof(size_t)); // Allocate size array for this level
        
        for(int j=0; j<n_line_size; j++){
            int idx1 = j*2;
            int idx2 = (j*2)+1;
            
            // Handle edge case when we have odd number of elements
            if (idx2 >= n_line_size * 2 && i > 1) {
                idx2 = idx1; // Use the same number twice
            }
            
            size_t s1 = s[i-1][idx1];
            size_t s2 = (idx2 < prev) ? s[i-1][idx2] : 0;
            
            // If s2 is 0, it means we reached the edge of odd number of elements
            if (s2 == 0) {
                // Just copy the number directly
                aoa[i][j] = (uint32_t*)malloc(s1 * sizeof(uint32_t));
                memcpy(aoa[i][j], aoa[i-1][idx1], s1 * sizeof(uint32_t));
                s[i][j] = s1;
                printf("copied down below for %d\n",j);
                continue;
            }
            
            // Check if this multiplication will fit in GPU memory
            size_t mem_needed = (s1 + s2 + s1 + s2) * sizeof(uint32_t);
            if (mem_needed > max_usable_mem) {
                printf("Warning: Multiplication at level %d, index %d too large for GPU memory\n", i, j);
                printf("Sizes: %zu and %zu units, memory needed: %zu MB\n", 
                       s1, s2, mem_needed/(1024*1024));
                
                // Fall back to CPU multiplication using GMP
                mpz_t n1, n2, result;
                mpz_init(n1);
                mpz_init(n2);
                mpz_init(result);
                
                // Convert uint32_t array back to mpz_t
                for (size_t k = 0; k < s1; k++) {
                    mpz_t temp;
                    mpz_init(temp);
                    mpz_set_ui(temp, aoa[i-1][idx1][s1-1-k]);
                    mpz_mul_2exp(temp, temp, 32*k);
                    mpz_add(n1, n1, temp);
                    mpz_clear(temp);
                }
                
                for (size_t k = 0; k < s2; k++) {
                    mpz_t temp;
                    mpz_init(temp);
                    mpz_set_ui(temp, aoa[i-1][idx2][s2-1-k]);
                    mpz_mul_2exp(temp, temp, 32*k);
                    mpz_add(n2, n2, temp);
                    mpz_clear(temp);
                }
                
                // Multiply using GMP
                mpz_mul(result, n1, n2);
                
                // Convert back to uint32_t array
                size_t result_size = (mpz_sizeinbase(result, 2) + 31) / 32;
                aoa[i][j] = (uint32_t*)malloc(result_size * sizeof(uint32_t));
                s[i][j] = result_size;
                
                mpz_t temp;
                mpz_init_set(temp, result);
                for (size_t k = 0; k < result_size; k++) {
                    aoa[i][j][k] = mpz_get_ui(temp);
                    mpz_fdiv_q_2exp(temp, temp, 32);
                }
                
                mpz_clear(temp);
                mpz_clear(n1);
                mpz_clear(n2);
                mpz_clear(result);
                
                continue;
            }
            
            // Allocate memory for the result (which can be up to s1+s2 in size)
            aoa[i][j] = (uint32_t*) malloc((s1+s2) * sizeof(uint32_t));
            s[i][j] = s1 + s2;  // Store size of the result
            
            uint32_t *d_num1, *d_num2, *d_result;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num1, s1 * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num2, s2 * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, (s1+s2) * sizeof(uint32_t)));

            CHECK_CUDA_ERROR(cudaMemcpy(d_num1, aoa[i-1][idx1], s1 * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_num2, aoa[i-1][idx2], s2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
            
            // Initialize result array to zeros
            CHECK_CUDA_ERROR(cudaMemset(d_result, 0, (s1+s2) * sizeof(uint32_t)));

            kernel<<<1,1>>>(d_num1, d_num2, s1, s2, d_result);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDA_ERROR(cudaMemcpy(aoa[i][j], d_result, (s1+s2) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaFree(d_num1));
            CHECK_CUDA_ERROR(cudaFree(d_num2));
            CHECK_CUDA_ERROR(cudaFree(d_result));

            char* num1 = uint32_array_to_hex(aoa[i-1][idx1], s1);
            char* num2 = uint32_array_to_hex(aoa[i-1][idx2], s2);
            char* result_hex = uint32_array_to_hex(aoa[i][j], s[i][j]);

            mpz_t g_num1;
            mpz_t g_num2;

            mpz_init_set_str(g_num1, num1, 16);
            mpz_init_set_str(g_num2, num2, 16);

            mpz_t mul;
            mpz_init(mul);
            mpz_mul(mul, g_num1, g_num2);

            
            char* gmp_hex_result = mpz_get_str(NULL, 16, mul);

            if (compare_strings(result_hex, gmp_hex_result)) {
                printf("The strings match (when compared from the end)! for %d\n",j);
            } else {
                printf("The strings do not match (when compared from the end)! for %d\n",j);
                FILE *file = fopen("output.txt", "a");
                if (file) {
                    fprintf(file, "num1: %s\n", num1);
                    fprintf(file, "num2: %s\n", num2);
                    fprintf(file, "\n");
                    fclose(file);
                }
            }
            
            // Free the allocated strings
            free(num1);
            free(num2);
            free(result_hex);
            free(gmp_hex_result);
            
            // Clear GMP variables
            mpz_clear(g_num1);
            mpz_clear(g_num2);
            mpz_clear(mul);
            
            printf("\n");
        }
        prev = n_line_size;
        n_line_size = (n_line_size+1)/2;
    }

    char* result_hex = uint32_array_to_hex(aoa[row-1][0], s[row-1][0]);
    printf("result Hex:%s \n",result_hex);

    free(result_hex);

    // Free memory
    for (int i = 0; i < line_size; i++) {
        mpz_clear(fileData[i]);
    }
    free(fileData);

    // Free allocated memory for aoa
    for (int i = 0; i < row; i++) {
        int elements = line_size / (1 << i);
        if (elements == 0) elements = 1; // At least one element
        
        for (int j = 0; j < elements; j++) {
            free(aoa[i][j]);
        }
        free(aoa[i]);
        free(s[i]);
    }
    free(aoa);
    free(s);

    return 0;
}