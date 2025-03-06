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

__global__ void kernel(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, uint32_t* d_result) {
  for (size_t i = 0; i < s1 + s2; i++) {
      d_result[i] = 0;
  }
  
  for (size_t i = 0; i < s1; i++) {
      uint64_t carry = 0;
      
      for (size_t j = 0; j < s2; j++) {
        size_t pos = i + j;
        uint64_t prod = (uint64_t)num1[i] * (uint64_t)num2[j] + d_result[pos] + carry;
          
        d_result[pos] = (uint32_t)(prod & 0xFFFFFFFF);
          
        carry = prod >> 32;
      }
      
      size_t idx = i + s2;
      while (carry > 0 && idx < (s1+s2)) {
          uint64_t sum = (uint64_t)d_result[idx] + carry;
          d_result[idx] = (uint32_t)(sum & 0xFFFFFFFF);
          carry = sum >> 32;
          idx++;  // Move forward, not backward
      }
  }
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
      sprintf(ptr, "%08X", arr[size - 1 - i]);
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
      // printf("Difference found: %c from str1 vs %c from str3\n",str1[length1 - 1 - i], str3[length2_actual - 1 - i]);
      FILE *file = fopen("output.txt", "a");
      if (file) {
        fprintf(file, "Difference found: %c from str1 vs %c from str3\n", str1[length1 - 1 - i], str3[length2_actual - 1 - i]);
        fclose(file);
      }
      flag = false;
    }
  }
  free(str3);
  return flag;
}

int main(){
  int line_size;
  mpz_t* fileData = readFile("./input-1000.txt", &line_size);

  long row = 0;
  for (int i = line_size; i >= 1; i = (i + 1) / 2) {
    row++;
    if (i == 1)
      break;
  }

  mpz_t ** array_of_arrays = (mpz_t ** ) malloc(row * sizeof(mpz_t * ));

  uint32_t*** aoa = (uint32_t***) malloc(row * sizeof(uint32_t**));
  size_t** s = (size_t**) malloc(row * sizeof(size_t*));

  int n_line_size = line_size;
  int prev = 0;

  for(int i=0 ; i<row ; i++){
    printf("level %d\n",i);
    array_of_arrays[i] = (mpz_t*) malloc(n_line_size * sizeof(mpz_t));

    for(int j=0; j<n_line_size; j++){
      mpz_init(array_of_arrays[i][j]);
    }

    if(i == 0){
      printf("0 exec\n");
        ////CPU////
      for(int j=0 ; j<n_line_size ; j++){
        mpz_set(array_of_arrays[i][j], fileData[j]);
      }

        ////GPU////

      aoa[i] = (uint32_t**) malloc(n_line_size * sizeof(uint32_t*));
      s[i] = (size_t*) malloc(n_line_size * sizeof(size_t));

      for (int k = 0; k < n_line_size; k++) {
        size_t s1 = ((mpz_sizeinbase(fileData[k], 2) + 31) / 32);
        s[0][k] = s1;
        aoa[0][k] = (uint32_t*) malloc(s1 * sizeof(uint32_t));
        mpz_t temp;
        mpz_init_set(temp, fileData[k]);
        for (size_t j = 0; j < s1; j++) {
          aoa[0][k][j] = (uint32_t) mpz_get_ui(temp);
          mpz_fdiv_q_2exp(temp, temp, 32);
        }
        mpz_clear(temp);
      }

      for(int j=0 ; j<n_line_size ; j++){
        char * self_result = uint32_array_to_hex(aoa[i][j],s[i][j]);
        char * gmp_result = mpz_get_str(NULL, 16, array_of_arrays[i][j]);

        if (compare_strings(self_result, gmp_result)) {
          printf("The strings match (when compared from the end)! for %d\n",j);
        } 
        else {
          printf("The strings do not match (when compared from the end)! for %d\n",j);
          FILE *file = fopen("output.txt", "a");
          if (file) {
            fprintf(file, "SELF: %s\n", self_result);
            fprintf(file, "GMP_: %s\n", gmp_result);
            fprintf(file, "\n");
            fclose(file);
          }
        }
      }
      
      prev = n_line_size;
      n_line_size = (n_line_size+1)/2;
      continue;
    }

    printf("mul execution\n");

      ////CPU////

    for(int j=0 ; j<n_line_size ; j++){
      if((j*2+1) < prev){
        mpz_mul(array_of_arrays[i][j], array_of_arrays[i-1][(j*2)], array_of_arrays[i-1][(j*2)+1]);
      } else {
        mpz_set(array_of_arrays[i][j], array_of_arrays[i-1][(j*2)]);
      }
    }

    ////GPU////

    aoa[i] = (uint32_t**) malloc(n_line_size * sizeof(uint32_t*));
    s[i] = (size_t*) malloc(n_line_size * sizeof(size_t));

    for(int j=0 ; j<n_line_size ; j++){
      if((j*2+1) >= prev){
        size_t s1 = s[i-1][j*2];
        aoa[i][j] = (uint32_t*)malloc(s1 * sizeof(uint32_t));
        memcpy(aoa[i][j], aoa[i-1][j*2], s1 * sizeof(uint32_t));
        s[i][j] = s1;
        printf("copied down below for %d\n",j);
        continue;
      } 
      else {
        size_t s1 = s[i-1][j*2];
        size_t s2 = s[i-1][((j*2)+1)];
        aoa[i][j] = (uint32_t*) malloc((s1+s2) * sizeof(uint32_t));
        s[i][j] = s1 + s2;
        uint32_t *d_num1, *d_num2, *d_result;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num1, s1 * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num2, s2 * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, (s1+s2) * sizeof(uint32_t)));
    
        CHECK_CUDA_ERROR(cudaMemcpy(d_num1, aoa[i-1][(j*2)], s1 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_num2, aoa[i-1][(j*2)+1], s2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
                
        // Initialize result array to zeros
        CHECK_CUDA_ERROR(cudaMemset(d_result, 0, (s1+s2) * sizeof(uint32_t)));
    
        kernel<<<1,1>>>(d_num1, d_num2, s1, s2, d_result);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
        CHECK_CUDA_ERROR(cudaGetLastError());
    
        CHECK_CUDA_ERROR(cudaMemcpy(aoa[i][j], d_result, (s1+s2) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
        CHECK_CUDA_ERROR(cudaFree(d_num1));
        CHECK_CUDA_ERROR(cudaFree(d_num2));
        CHECK_CUDA_ERROR(cudaFree(d_result));
      }
    }

    for(int j=0 ; j<n_line_size ; j++){
      char * self_result = uint32_array_to_hex(aoa[i][j],s[i][j]);
      char * gmp_result = mpz_get_str(NULL, 16, array_of_arrays[i][j]);

      if (compare_strings(self_result, gmp_result)) {
        printf("The strings match (when compared from the end)! for %d\n",j);
      } 
      else {
        printf("The strings do not match (when compared from the end)! for %d\n",j);
        FILE *file = fopen("output.txt", "a");
        if (file) {
          fprintf(file, "SELF: %s\n", self_result);
          fprintf(file, "GMP_: %s\n", gmp_result);
          fprintf(file, "\n");
          fclose(file);
        }
      }
    }

    prev = n_line_size;
    n_line_size = (n_line_size+1)/2;
  }
  // mpz_out_str(stdout, 16, array_of_arrays[row-1][0]);
  printf("\n");
}