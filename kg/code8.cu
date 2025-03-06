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

// CUDA kernel for single multiplication
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
}

// CUDA kernel that processes multiple multiplications in parallel
__global__ void batch_multiply_kernel(uint32_t** inputs, size_t* sizes, uint32_t** results, 
                                    int n_pairs) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= n_pairs)
        return;
    
    int idx1 = pair_idx * 2;
    int idx2 = (pair_idx * 2) + 1;
    
    size_t s1 = sizes[idx1];
    size_t s2 = sizes[idx2];
    
    uint32_t* num1 = inputs[idx1];
    uint32_t* num2 = inputs[idx2];
    uint32_t* result = results[pair_idx];
    
    // Initialize result to zero
    for (size_t i = 0; i < s1 + s2; i++) {
        result[i] = 0;
    }
    
    // Standard long multiplication algorithm
    for (size_t i = 0; i < s1; i++) {
        uint64_t carry = 0;
        
        for (size_t j = 0; j < s2; j++) {
            // Calculate position in result array
            size_t pos = (s1 + s2) - 1 - (i + j);
            
            // Calculate the product and add to existing value with carry
            uint64_t prod = (uint64_t)num1[s1-1-i] * (uint64_t)num2[s2-1-j] + 
                           result[pos] + carry;
            
            // Store the lower 32 bits
            result[pos] = (uint32_t)(prod & 0xFFFFFFFF);
            
            // Carry the upper bits
            carry = prod >> 32;
        }
        
        // Handle remaining carry
        size_t idx = (s1+s2) - 1 - (i + s2);
        while (carry > 0 && idx < (s1+s2)) {
            uint64_t sum = (uint64_t)result[idx] + carry;
            result[idx] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
            if (idx == 0) break;
            idx--;
        }
    }
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
  
    mpz_t * array = (mpz_t * ) malloc(line_count * sizeof(mpz_t));
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

int main() {
    int line_size;
    mpz_t* fileData = readFile("./input-1000.txt", &line_size);
    if (!fileData) {
        fprintf(stderr, "Failed to read input file\n");
        return EXIT_FAILURE;
    }

    // Calculate number of levels in the tree
    long row = 0;
    for (int i = line_size; i >= 1; i = (i + 1) / 2) {
        row++;
        if (i == 1)
            break;
    }

    // Get GPU memory information
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory: %zu MB free out of %zu MB total\n", 
           free_mem/(1024*1024), total_mem/(1024*1024));
    
    // Leave some buffer for other operations
    size_t max_usable_mem = free_mem * 0.9;  // Use only 90% of available memory

    uint32_t*** aoa = (uint32_t***) malloc(row * sizeof(uint32_t**));
    size_t** s = (size_t**) malloc(row * sizeof(size_t*));
    
    // Initialize first level
    aoa[0] = (uint32_t**) malloc(line_size * sizeof(uint32_t*));
    s[0] = (size_t*) malloc(line_size * sizeof(size_t));

    // Convert mpz_t numbers to uint32_t arrays
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

    int prev_line_size = line_size;
    
    // Process each level
    for (int i = 1; i < row; i++) {
        int n_line_size = (prev_line_size + 1) / 2;
        printf("Running level %d with %d pairs to multiply\n", i, n_line_size);
        
        // Allocate arrays for this level
        aoa[i] = (uint32_t**) malloc(n_line_size * sizeof(uint32_t*));
        s[i] = (size_t*) malloc(n_line_size * sizeof(size_t));
        
        // Pre-calculate sizes and memory requirements
        size_t total_mem_needed = 0;
        
        for (int j = 0; j < n_line_size; j++) {
            int idx1 = j * 2;
            int idx2 = (j * 2) + 1;
            
            // Handle odd number of elements
            if (idx2 >= prev_line_size) {
                // For odd element, just copy the size
                s[i][j] = s[i-1][idx1];
            } else {
                // For pair, size will be sum of operand sizes
                s[i][j] = s[i-1][idx1] + s[i-1][idx2];  
            }
            
            // Add memory needed for this operation
            if (idx2 >= prev_line_size) {
                total_mem_needed += s[i][j] * sizeof(uint32_t);
            } else {
                total_mem_needed += (s[i-1][idx1] + s[i-1][idx2] + s[i][j]) * sizeof(uint32_t);
            }
        }
        
        // Add overhead for pointers and other structures
        total_mem_needed += (prev_line_size * sizeof(uint32_t*)) + 
                           (n_line_size * sizeof(uint32_t*)) + 
                           (prev_line_size * sizeof(size_t));
        
        printf("Memory needed for level %d: %zu MB\n", i, total_mem_needed/(1024*1024));
        
        // Check if we have enough GPU memory for batch processing
        if (total_mem_needed > max_usable_mem) {
            printf("Not enough GPU memory for batch processing. Falling back to sequential.\n");
            
            // Sequential processing - one multiplication at a time
            for (int j = 0; j < n_line_size; j++) {
                int idx1 = j * 2;
                int idx2 = (j * 2) + 1;
                
                // Handle last element for odd number of elements
                if (idx2 >= prev_line_size) {
                    // Just copy the element
                    size_t s1 = s[i-1][idx1];
                    aoa[i][j] = (uint32_t*) malloc(s1 * sizeof(uint32_t));
                    memcpy(aoa[i][j], aoa[i-1][idx1], s1 * sizeof(uint32_t));
                    s[i][j] = s1;
                    continue;
                }
                
                size_t s1 = s[i-1][idx1];
                size_t s2 = s[i-1][idx2];
                
                // Check if this individual multiplication fits in GPU memory
                size_t mem_needed = (s1 + s2 + (s1+s2)) * sizeof(uint32_t);
                aoa[i][j] = (uint32_t*) malloc((s1+s2) * sizeof(uint32_t));
                
                if (mem_needed > max_usable_mem / 4) {
                    // Too large even for individual GPU computation, use GMP on CPU
                    printf("Pair %d: Using GMP for large multiplication\n", j);
                    
                    // Convert uint32_t arrays to mpz_t
                    mpz_t n1, n2, result;
                    mpz_init(n1);
                    mpz_init(n2);
                    mpz_init(result);
                    
                    // Build first number
                    for (size_t k = 0; k < s1; k++) {
                        mpz_t temp;
                        mpz_init(temp);
                        mpz_set_ui(temp, aoa[i-1][idx1][s1-1-k]);
                        mpz_mul_2exp(temp, temp, 32*k);
                        mpz_add(n1, n1, temp);
                        mpz_clear(temp);
                    }
                    
                    // Build second number
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
                    
                    // Convert result back to uint32_t array
                    mpz_t temp;
                    mpz_init_set(temp, result);
                    for (size_t k = 0; k < s1+s2; k++) {
                        aoa[i][j][k] = mpz_get_ui(temp);
                        mpz_fdiv_q_2exp(temp, temp, 32);
                    }
                    
                    mpz_clear(temp);
                    mpz_clear(n1);
                    mpz_clear(n2);
                    mpz_clear(result);
                } else {
                    // Use GPU for this multiplication
                    uint32_t *d_num1, *d_num2, *d_result;
                    
                    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num1, s1 * sizeof(uint32_t)));
                    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num2, s2 * sizeof(uint32_t)));
                    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, (s1+s2) * sizeof(uint32_t)));
                    
                    CHECK_CUDA_ERROR(cudaMemcpy(d_num1, aoa[i-1][idx1], s1 * sizeof(uint32_t), cudaMemcpyHostToDevice));
                    CHECK_CUDA_ERROR(cudaMemcpy(d_num2, aoa[i-1][idx2], s2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
                    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, (s1+s2) * sizeof(uint32_t)));
                    
                    // Launch the single multiplication kernel
                    kernel<<<1, 1>>>(d_num1, d_num2, s1, s2, d_result);
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    
                    // Copy result back to host
                    CHECK_CUDA_ERROR(cudaMemcpy(aoa[i][j], d_result, (s1+s2) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    
                    // Free GPU memory
                    CHECK_CUDA_ERROR(cudaFree(d_num1));
                    CHECK_CUDA_ERROR(cudaFree(d_num2));
                    CHECK_CUDA_ERROR(cudaFree(d_result));
                }
            }
        } else {
            // Batch processing - use GPU to process multiple multiplications at once
            printf("Using batch processing for level %d\n", i);
            
            // Count the actual number of pairs to multiply
            int actual_pairs = 0;
            for (int j = 0; j < n_line_size; j++) {
                int idx2 = (j * 2) + 1;
                if (idx2 < prev_line_size) {
                    actual_pairs++;
                }
            }
            
            // Create arrays to store device pointers
            uint32_t **d_inputs, **d_results;
            uint32_t **h_d_inputs = (uint32_t**)malloc(prev_line_size * sizeof(uint32_t*));
            uint32_t **h_d_results = (uint32_t**)malloc(actual_pairs * sizeof(uint32_t*));
            size_t *d_sizes, *h_sizes = (size_t*)malloc(prev_line_size * sizeof(size_t));
            
            // Allocate device memory for array of pointers
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_inputs, prev_line_size * sizeof(uint32_t*)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_results, actual_pairs * sizeof(uint32_t*)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sizes, prev_line_size * sizeof(size_t)));
            
            // Copy all input arrays to device and store pointers
            for (int j = 0; j < prev_line_size; j++) {
                uint32_t *d_array;
                size_t size = s[i-1][j];
                h_sizes[j] = size;
                
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_array, size * sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemcpy(d_array, aoa[i-1][j], size * sizeof(uint32_t), 
                                          cudaMemcpyHostToDevice));
                h_d_inputs[j] = d_array;
            }
            
            // Create result arrays on device
            int pair_idx = 0;
            for (int j = 0; j < n_line_size; j++) {
                int idx1 = j * 2;
                int idx2 = (j * 2) + 1;
                
                // Skip if this is the last element in an odd-length list
                if (idx2 >= prev_line_size) {
                    // Just copy the element directly
                    size_t s1 = s[i-1][idx1];
                    aoa[i][j] = (uint32_t*) malloc(s1 * sizeof(uint32_t));
                    memcpy(aoa[i][j], aoa[i-1][idx1], s1 * sizeof(uint32_t));
                    s[i][j] = s1;
                    continue;
                }
                
                // Allocate memory for result
                size_t s1 = s[i-1][idx1];
                size_t s2 = s[i-1][idx2];
                size_t result_size = s1 + s2;
                
                aoa[i][j] = (uint32_t*) malloc(result_size * sizeof(uint32_t));
                s[i][j] = result_size;
                
                uint32_t *d_result;
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, result_size * sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(d_result, 0, result_size * sizeof(uint32_t)));
                h_d_results[pair_idx++] = d_result;
            }
            
            // Copy arrays of pointers to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_inputs, h_d_inputs, prev_line_size * sizeof(uint32_t*), 
                                      cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_results, h_d_results, actual_pairs * sizeof(uint32_t*), 
                                      cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_sizes, h_sizes, prev_line_size * sizeof(size_t), 
                                      cudaMemcpyHostToDevice));
            
            // Launch kernel with multiple threads
            int threadsPerBlock = 256;
            int blocksPerGrid = (actual_pairs + threadsPerBlock - 1) / threadsPerBlock;
            
            printf("Launching kernel with %d blocks, %d threads per block for %d pairs\n", 
                   blocksPerGrid, threadsPerBlock, actual_pairs);
            
            batch_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_inputs, d_sizes, d_results, actual_pairs
            );
            
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Copy results back to host
            pair_idx = 0;
            for (int j = 0; j < n_line_size; j++) {
                int idx2 = (j * 2) + 1;
                
                // Skip if this was a single element (not a pair)
                if (idx2 >= prev_line_size) {
                    continue;
                }
                
                CHECK_CUDA_ERROR(cudaMemcpy(aoa[i][j], h_d_results[pair_idx++], 
                                         s[i][j] * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            }
            
            // Free device memory
            for (int j = 0; j < prev_line_size; j++) {
                CHECK_CUDA_ERROR(cudaFree(h_d_inputs[j]));
            }
            
            for (int j = 0; j < actual_pairs; j++) {
                CHECK_CUDA_ERROR(cudaFree(h_d_results[j]));
            }
            
            CHECK_CUDA_ERROR(cudaFree(d_inputs));
            CHECK_CUDA_ERROR(cudaFree(d_results));
            CHECK_CUDA_ERROR(cudaFree(d_sizes));
            
            free(h_d_inputs);
            free(h_d_results);
            free(h_sizes);
        }
        
        prev_line_size = n_line_size;
    }
    
    // Print the final result if available
    if (row > 0) {
        printf("Final result has %zu 32-bit words\n", s[row-1][0]);
        
        // You can add code here to verify the result if needed
        char* result_hex = uint32_array_to_hex(aoa[row-1][0], s[row-1][0]);
        // if (result_hex) {
        //     printf("Result prefix: %.64s...\n", result_hex);
        //     free(result_hex);
        // }
        long long row_size[row];
        row_size[row-1] = line_size;
      
        mpz_t ** array_of_arrays = (mpz_t ** ) malloc(row * sizeof(mpz_t * ));
        
        int ls = ceil(line_size/2.0);
      
        for (int i=ls, count=0, prev=line_size ; i>=1 ; i=ceil(i/2.0), count+=1) {
          array_of_arrays[count] = (mpz_t * ) malloc(i * sizeof(mpz_t));
          //printf("count is : %d i allocated is: %d prev is: %d\n", count, i, prev);
      
          for (int j = 0; j < i; j++) {
            mpz_init(array_of_arrays[count][j]);
          }
      
          if (i == ls) {
            for (int j = 0, k = 0; j < i; j += 1, k += 2) {
              if (k == prev - 1) {
                mpz_set(array_of_arrays[count][j], fileData[k]);
              } else {
                mpz_mul(array_of_arrays[count][j], fileData[k], fileData[k + 1]);
              }
            }
          } else if (i == 1) {
            mpz_mul(array_of_arrays[count][0], array_of_arrays[count - 1][0], array_of_arrays[count - 1][1]);
              row_size[row-2-count] = i;
            break;
          } else {
            for (int j = 0, k = 0; j < i; j += 1, k += 2) {
              if (k == prev - 1) {
                mpz_set(array_of_arrays[count][j], array_of_arrays[count - 1][k]);
              } else {
                mpz_mul(array_of_arrays[count][j], array_of_arrays[count - 1][k], array_of_arrays[count - 1][k + 1]);
              }
            }
          }
          prev = i;
          row_size[row-2-count] = i;
        }
        printf("[MAIN] Product Tree constructed\n");

        char* gmp_hex_result = mpz_get_str(NULL, 16, array_of_arrays[row-2][0]);

        if (compare_strings(result_hex, gmp_hex_result)) {
            printf("The strings match (when compared from the end)!\n");
        } else {
            printf("The strings do not match (when compared from the end)!\n");
        }
        printf("\n");

        free(gmp_hex_result);
    }

    // Free memory
    for (int i = 0; i < line_size; i++) {
        mpz_clear(fileData[i]);
    }
    free(fileData);

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