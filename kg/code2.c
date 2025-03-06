#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <gmp.h>
#include <stdbool.h>
#include <ctype.h>
#include <cuda.h>
#include <math.h>

uint32_t* sum(uint32_t* a, uint32_t* b, size_t sa, size_t sb, size_t* res) {
    size_t max = (sa > sb) ? sa : sb;
    uint64_t carry = 0;
    uint64_t tmp_sum;
    
    uint32_t* temp_res = (uint32_t*) malloc(max * sizeof(uint32_t));
    if (temp_res == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < max; i++) {
        uint32_t a_num = (i < sa) ? a[sa - 1 - i] : 0;
        uint32_t b_num = (i < sb) ? b[sb - 1 - i] : 0;
        tmp_sum = (uint64_t)a_num + (uint64_t)b_num + carry;
        temp_res[max - 1 - i] = (uint32_t)(tmp_sum & 0xFFFFFFFF);
        carry = tmp_sum >> 32;
    }
    
    if (carry) {
        uint32_t* result = (uint32_t*) malloc((max + 1) * sizeof(uint32_t));
        if (result == NULL) {
            free(temp_res);
            return NULL;
        }
        result[0] = (uint32_t) carry;
        for (size_t i = 0; i < max; i++) {
            result[i + 1] = temp_res[i];
        }
        free(temp_res);
        *res = max + 1;
        return result;
    } else {
        *res = max;
        return temp_res;
    }
}

uint32_t* diff(uint32_t* a, uint32_t* b, size_t sa, size_t sb, size_t* res) {
    size_t max = (sa > sb) ? sa : sb;
    uint32_t* sub_str = (uint32_t*) malloc(max * sizeof(uint32_t));
    uint32_t borrow = 0;
    
    for (int i = 0; i < max; i++) {
        uint32_t a_num = (i < sa) ? a[sa - 1 - i] : 0;
        uint32_t b_num = (i < sb) ? b[sb - 1 - i] : 0;
        
        uint32_t diff_val;
        if (a_num < b_num + borrow) {
            diff_val = a_num + ((uint64_t)1 << 32) - b_num - borrow;
            borrow = 1;
        } else {
            diff_val = a_num - b_num - borrow;
            borrow = 0;
        }
        sub_str[max -1 -i] = diff_val;
    }
    
    *res = max;
    return sub_str;
}

uint32_t* multiply(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, size_t* res){
    size_t max = s1 > s2 ? s1 : s2;
    // printf("max is: %lu\n", (unsigned long)max);
    

    ///////////////////////////////////////
    // for (size_t i = 0; i < s1; i++) {
    //     printf("%08x ", num1[i]);
    // }
    // printf("\n");
    // for (size_t i = 0; i < s1; i++) {
    //     printf("%u ", num1[i]);
    // }
    // printf("\n");
    
    // for (size_t i = 0; i < s2; i++) {
    //     printf("%08x ", num2[i]);
    // }
    // printf("\n");
    // for (size_t i = 0; i < s2; i++) {
    //     printf("%u ", num2[i]);
    // }
    // printf("\n");
    /////////////////////////////////////////
    
    if(max == 1){
        *res = 2;
        uint32_t* mul = (uint32_t*) malloc(2 * sizeof(uint32_t));
        if(s1 == 0 || s2 == 0){
            mul[0] = 0;
            mul[1] = 0;
            return mul;
        }
        // printf("if is true\n");
        uint64_t im_mul = (uint64_t) num1[0] * (uint64_t)num2[0];
        mul[1] = (uint32_t) (im_mul & 0xFFFFFFFF);
        mul[0] = (uint32_t) (im_mul >> 32);
        return mul;
    }

    max = (max % 2 == 0) ? max : max + 1;

    size_t pad1 = max-s1;
    size_t pad2 = max-s2;
    // printf("pad1 : %lu\n",(unsigned long)pad1);
    // printf("pad2 : %lu\n",(unsigned long)pad2);

    uint32_t* p_num1 = (uint32_t*) malloc(max * sizeof(uint32_t));
    uint32_t* p_num2 = (uint32_t*) malloc(max * sizeof(uint32_t));

    for(int i=0 ; i<pad1 ; i++){
        p_num1[i] = 0;
    }
    for(int i=pad1 ; i<max ; i++){
        p_num1[i] = num1[i-pad1];
    }
    
    for(int i=0 ; i<pad2 ; i++){
        p_num2[i] = 0;
    }
    for(int i=pad2 ; i<max ; i++){
        p_num2[i] = num2[i-pad2];
    }

    ///////////////////////////////////////
    // for (size_t i = 0; i < max; i++) {
    //     printf("%08x ", p_num1[i]);
    // }
    // printf("\n");
    // for (size_t i = 0; i < max; i++) {
    //     printf("%u ", p_num1[i]);
    // }
    // printf("\n");
    
    // for (size_t i = 0; i < max; i++) {
    //     printf("%08x ", p_num2[i]);
    // }
    // printf("\n");
    // for (size_t i = 0; i < max; i++) {
    //     printf("%u ", p_num2[i]);
    // }
    // printf("\n");
    /////////////////////////////////////////

    uint32_t* a = (uint32_t*) malloc((max/2) * sizeof(uint32_t));
    memcpy(a, p_num1, max/2  * sizeof(uint32_t));
    uint32_t* b = (uint32_t*) malloc((max/2) * sizeof(uint32_t));
    memcpy(b, p_num1+(max/2), max/2 * sizeof(uint32_t));
    
    uint32_t* c = (uint32_t*) malloc((max/2) * sizeof(uint32_t));
    memcpy(c, p_num2, max/2 * sizeof(uint32_t));
    uint32_t* d = (uint32_t*) malloc((max/2) * sizeof(uint32_t));
    memcpy(d, p_num2+(max/2), max/2 * sizeof(uint32_t));

    
    /////////////////////////////////////////////
    // printf("a as hex:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%08x ", a[i]);
    // }
    // printf("\n");

    // printf("a as decimal:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%u ",a[i]);
    // }
    // printf("\n");

    // printf("b as hex:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%08x ", b[i]);
    // }
    // printf("\n");

    // printf("b as decimal:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%u ",b[i]);
    // }
    // printf("\n");
    
    // printf("c as hex:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%08x ", c[i]);
    // }
    // printf("\n");

    // printf("c as decimal:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%u ",c[i]);
    // }
    // printf("\n");

    // printf("d as hex:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%08x ", d[i]);
    // }
    // printf("\n");

    // printf("d as decimal:\n");
    // for (size_t i = 0; i < max/2; i++) {
    //     printf("%u ",d[i]);
    // }
    // printf("\n");
    /////////////////////////////////////////////////////

    size_t s_ac, s_bd;
    uint32_t* ac = multiply(a, c, max/2, max/2, &s_ac);
    uint32_t* bd = multiply(b, d, max/2, max/2, &s_bd);
    

    // printf("ac as hex: ");
    // for (size_t i = 0; i < s_ac; i++) {
    //     printf("%08x ", ac[i]);
    // }
    // printf(" - ");

    // printf("ac as decimal: ");
    // for (size_t i = 0; i < s_ac; i++) {
    //     printf("%u ",ac[i]);
    // }
    // printf("\n");

    // printf("bd as hex: ");
    // for (size_t i = 0; i < s_bd; i++) {
    //     printf("%08x ", bd[i]);
    // }
    // printf(" - ");

    // printf("bd as decimal: ");
    // for (size_t i = 0; i < s_bd; i++) {
    //     printf("%u ",bd[i]);
    // }
    // printf("\n");

    size_t s_a_plus_b, s_c_plus_d; 
    uint32_t* a_plus_b = sum(a, b ,max/2 ,max/2, &s_a_plus_b);
    uint32_t* c_plus_d = sum(c, d ,max/2 ,max/2, &s_c_plus_d);
    
    // printf("a_plus_b as hex: ");
    // for (size_t i = 0; i < s_a_plus_b; i++) {
    //     printf("%08x ", a_plus_b[i]);
    // }
    // printf(" - ");

    // printf("a_plus_b as decimal: ");
    // for (size_t i = 0; i < s_a_plus_b; i++) {
    //     printf("%u ",a_plus_b[i]);
    // }
    // printf("\n");

    // printf("c_plus_d as hex: ");
    // for (size_t i = 0; i < s_c_plus_d; i++) {
    //     printf("%08x ", c_plus_d[i]);
    // }
    // printf(" - ");

    // printf("c_plus_d as decimal: ");
    // for (size_t i = 0; i < s_c_plus_d; i++) {
    //     printf("%u ",c_plus_d[i]);
    // }
    // printf("\n");
    
    // printf("\n");

    size_t s_a_plus_b_times_c_plus_d;
    uint32_t* a_plus_b_times_c_plus_d = multiply(a_plus_b, c_plus_d, s_a_plus_b, s_c_plus_d, &s_a_plus_b_times_c_plus_d);

    size_t s_diff1, s_diff2;
    uint32_t* diff1 = diff(a_plus_b_times_c_plus_d, ac, s_a_plus_b_times_c_plus_d, s_ac, &s_diff1);
    uint32_t* diff2 = diff(diff1, bd, s_diff1, s_bd, &s_diff2);

    uint32_t* shifted_ac = (uint32_t*) malloc( (s_ac+max) * sizeof(uint32_t));

    for( size_t i=0 ; i<(s_ac+max) ; i++){
        shifted_ac[i] = (i<s_ac) ? ac[i] : 0;
    }

    uint32_t* shifted_diff2 = (uint32_t*) malloc( (s_diff2+(max/2)) * sizeof(uint32_t));

    for( size_t i=0 ; i<(s_diff2+(max/2)) ; i++){
        shifted_diff2[i] = (i<s_diff2) ? diff2[i] : 0;
    }

    size_t s_sum1, s_sum2;
    uint32_t* sum1 = sum( shifted_ac, shifted_diff2, (s_ac+max), (s_diff2+(max/2)), &s_sum1);
    uint32_t* sum2 = sum( sum1, bd, s_sum1, s_bd, &s_sum2);

    *res = s_sum2;

    free(p_num1);
    free(p_num2);
    free(a);
    free(b);
    free(c);
    free(d);
    free(ac);
    free(bd);
    free(a_plus_b);
    free(c_plus_d);
    free(a_plus_b_times_c_plus_d);
    free(diff1);
    free(diff2);
    free(sum1);
    free(shifted_ac);
    free(shifted_diff2);
    return sum2;
}

void kernel(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, uint32_t* d_result) {
    size_t s_res;
    uint32_t* final = multiply(num1, num2, s1, s2, &s_res);

    // printf("final mul as hex (GPU):\n");
    // for (size_t i = 0; i < s_res; i++) {
    //     printf("%08x ", final[i]);
    // }
    // printf("\n");

    // printf("final mul as decimal (GPU):\n");
    // for (size_t i = 0; i < s_res; i++) {
    //     printf("%u ",final[i]);
    // }
    // printf("\n");

    // if(s_res > (s1+s2)){
    //     printf("greater than expected size diff of : %lu\n", (unsigned long)(s_res-(s1+s2)));
    // }
    for(int i=0; i<(s1+s2); i++){
        d_result[(s1+s2) -1 -i] = (i<s_res) ? final[s_res-1-i] : 0 ;
    }
    free(final);

    return;
}

char* gen(long long length) {
    char* hex_string = (char*)malloc((length + 1) * sizeof(char));
    if (hex_string == NULL) {
        return NULL;
    }
    
    const char hex_chars[] = "0123456789ABCDEF";
    for (long long i = 0; i < length; i++) {
        long long random_index = rand() % 16;
        hex_string[i] = hex_chars[random_index];
    }
    hex_string[length] = '\0';
    return hex_string;
}

char* uint32_array_to_hex(uint32_t* arr, size_t size) {
    if (!arr || size == 0) return NULL;
    
    // Each uint32_t can take up to 8 hex characters, plus null terminator
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
    *ptr = '\0'; // Null-terminate the string
    
    return hex_str;
}

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
    
    for (size_t i = 0; i < min_length; i++) {
        if (str1[length1 - 1 - i] != str3[length2_actual - 1 - i]) {
            printf("Difference found: %c from str1 vs %c from str3\n",
                   str1[length1 - 1 - i], str3[length2_actual - 1 - i]);
            free(str3);
            return false;
        }
    }
    free(str3);
    return true;
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
    struct timespec start, end1, end2, end3, end4;

    clock_gettime(CLOCK_REALTIME, &start);

    int line_size;
    mpz_t *fileData = readFile("./input-100k.txt", &line_size);

    clock_gettime(CLOCK_REALTIME, &end1);
    printf("[MAIN] Reading file done by time: %f seconds\n", (end1.tv_sec - start.tv_sec) + (end1.tv_nsec - start.tv_nsec) / 1.0e9);

    // Determine total number of rows for the tree.
    int total_rows = 1;
    int temp = line_size;
    while (temp > 1) {
        total_rows++;
        temp = (temp + 1) / 2;
    }

    uint32_t*** aoaoa = (uint32_t***) malloc(total_rows * sizeof(uint32_t**));
    size_t** sizes = (size_t**) malloc(total_rows * sizeof(size_t*));

    // Row 0: store each fileData converted to uint32_t arrays.
    aoaoa[0] = (uint32_t**) malloc(line_size * sizeof(uint32_t*));
    sizes[0] = (size_t*) malloc(line_size * sizeof(size_t));
    for (int i = 0; i < line_size; i++) {
        size_t bit_size = mpz_sizeinbase(fileData[i], 2);
        size_t numSize = (bit_size + 31) / 32;
        aoaoa[0][i] = (uint32_t*) malloc(numSize * sizeof(uint32_t));
        sizes[0][i] = numSize;
        mpz_t temp_mpz;
        mpz_init(temp_mpz);
        mpz_set(temp_mpz, fileData[i]);
        for (size_t j = 0; j < numSize; j++) {
            aoaoa[0][i][numSize - 1 - j] = (uint32_t) mpz_get_ui(temp_mpz);
            mpz_fdiv_q_2exp(temp_mpz, temp_mpz, 32);
        }
        mpz_clear(temp_mpz);
    }
    clock_gettime(CLOCK_REALTIME, &end2);
    printf("[MAIN] fileData Prepped for multiplicaton by time: %f seconds\n", (end2.tv_sec - start.tv_sec) + (end2.tv_nsec - start.tv_nsec) / 1.0e9);

    // Build the tree by combining pairs using kernel().
    int parent_count = line_size;
    for (int i = 1; i < total_rows; i++) {
        printf("level %d\n",i);
        int child_count = (parent_count + 1) / 2;
        aoaoa[i] = (uint32_t**) malloc(child_count * sizeof(uint32_t*));
        sizes[i] = (size_t*) malloc(child_count * sizeof(size_t));
        for (int j = 0; j < child_count; j++) {
            int idx_left = j * 2;
            if (idx_left + 1 < parent_count) {
                size_t combined_size = sizes[i - 1][idx_left] + sizes[i - 1][idx_left + 1];
                aoaoa[i][j] = (uint32_t*) malloc(combined_size * sizeof(uint32_t));
                kernel(aoaoa[i - 1][idx_left], aoaoa[i - 1][idx_left + 1],
                       sizes[i - 1][idx_left], sizes[i - 1][idx_left + 1],
                       aoaoa[i][j]);
                sizes[i][j] = combined_size;
            } else {
                size_t copy_size = sizes[i - 1][idx_left];
                aoaoa[i][j] = (uint32_t*) malloc(copy_size * sizeof(uint32_t));
                memcpy(aoaoa[i][j], aoaoa[i - 1][idx_left], copy_size * sizeof(uint32_t));
                sizes[i][j] = copy_size;
            }
        }
        parent_count = child_count;
    }
    clock_gettime(CLOCK_REALTIME, &end3);
    printf("[MAIN] SELF multiplicaton done by time: %f seconds\n", (end3.tv_sec - start.tv_sec) + (end3.tv_nsec - start.tv_nsec) / 1.0e9);

    long long row_size[total_rows];
    row_size[total_rows-1] = line_size;

    mpz_t ** array_of_arrays = (mpz_t ** ) malloc(total_rows * sizeof(mpz_t * ));
    
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
            row_size[total_rows-2-count] = i;
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
        row_size[total_rows-2-count] = i;
    }
    clock_gettime(CLOCK_REALTIME, &end4);
    printf("[MAIN] GMP_ multiplicaton done by time: %f seconds\n", (end4.tv_sec - start.tv_sec) + (end4.tv_nsec - start.tv_nsec) / 1.0e9);

    // Print the final result in hexadecimal using uint32_array_to_hex
    char* final_hex = uint32_array_to_hex(aoaoa[total_rows - 1][0], sizes[total_rows - 1][0]);

    char* gmp_hex_result = mpz_get_str(NULL, 16, array_of_arrays[total_rows-2][0]);

    if (compare_strings(final_hex, gmp_hex_result)) {
        printf("The strings match (when compared from the end)!\n");
    } else {
        printf("The strings do not match (when compared from the end)!\n");
    }
    printf("\n");

    // Free allocated memory.
    int* counts = (int*) malloc(total_rows * sizeof(int));
    counts[0] = line_size;
    for (int i = 1; i < total_rows; i++) {
        counts[i] = (counts[i - 1] + 1) / 2;
    }
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < counts[i]; j++) {
            free(aoaoa[i][j]);
        }
        free(aoaoa[i]);
        free(sizes[i]);
    }
    free(aoaoa);
    free(sizes);
    for (int i = 0; i < line_size; i++) {
        mpz_clear(fileData[i]);
    }
    free(fileData);
    free(counts);
    free(final_hex);
    return 0;
}