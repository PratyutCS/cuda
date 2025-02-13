#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>

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


__global__ void multilply(){

}

void print_binary(const unsigned char *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        for (int bit = 7; bit >= 0; bit--) {
            printf("%d", (data[i] >> bit) & 1);
        }
        printf(" ");  // Space for readability
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

    for (int i = 0; i < line_size; i++) {
        binaryArray[i] = (unsigned char *)mpz_export(NULL, &binarySizes[i], 1, 1, 1, 0, fileData[i]);
    }

    for (int i = 0; i < 1; i++) {
        printf("Binary output for line %d (%zu bytes):\n", i + 1, binarySizes[i]);
        gmp_printf("Hexadecimal value is: %ZX\n",fileData[i]);
        print_binary(binaryArray[i], binarySizes[i]);
    }

    for (int i = 0; i < line_size; i++) {
        free(binaryArray[i]);  
        mpz_clear(fileData[i]);
    }
    free(binaryArray);
    free(binarySizes);
    free(fileData);

    return 0;
}
