#include <stdio.h>
#include <gmp.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void print(){
    printf("works\n");
    return;
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

int main(){
    int line_size;
    mpz_t * fileData = readFile("./input-1000.txt", & line_size);

    uint32_t** aoa = (uint32_t**) malloc(line_size * (uint32_t*));

    print<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}