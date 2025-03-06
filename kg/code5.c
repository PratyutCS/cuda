#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <gmp.h>
#include <stdbool.h>
#include <ctype.h>
#include <cuda.h>

// Direct schoolbook multiplication - more reliable for all cases
uint32_t* schoolbook_multiply(uint32_t* a, uint32_t* b, size_t sa, size_t sb, size_t* res) {
    if (sa == 0 || sb == 0) {
        uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t));
        if (result) {
            result[0] = 0;
            *res = 1;
        } else {
            *res = 0;
        }
        return result;
    }
    
    // Allocate space for the result
    size_t result_size = sa + sb;
    uint32_t* result = (uint32_t*)calloc(result_size, sizeof(uint32_t));
    if (!result) {
        *res = 0;
        return NULL;
    }
    
    // Standard long multiplication algorithm
    for (size_t i = 0; i < sa; i++) {
        uint64_t carry = 0;
        
        for (size_t j = 0; j < sb; j++) {
            // Calculate the product and add to existing value with carry
            uint64_t prod = (uint64_t)a[sa-1-i] * (uint64_t)b[sb-1-j] + 
                           result[result_size-1-(i+j)] + carry;
            
            // Store the lower 32 bits
            result[result_size-1-(i+j)] = (uint32_t)(prod & 0xFFFFFFFF);
            
            // Carry the upper bits
            carry = prod >> 32;
        }
        
        // Handle remaining carry
        size_t idx = result_size-1-(i+sb);
        while (carry > 0 && idx < result_size) {
            uint64_t sum = (uint64_t)result[idx] + carry;
            result[idx] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
            if (idx == 0) break;
            idx--;
        }
    }
    
    // Remove leading zeros
    while (result_size > 1 && result[0] == 0) {
        memmove(result, result + 1, (result_size - 1) * sizeof(uint32_t));
        result_size--;
    }
    
    *res = result_size;
    return result;
}

// Main wrapper function
void kernel(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, uint32_t* d_result) {
    size_t s_res;
    uint32_t* final = schoolbook_multiply(num1, num2, s1, s2, &s_res);
    
    if (!final) {
        memset(d_result, 0, (s1+s2) * sizeof(uint32_t));
        return;
    }
    
    // Copy result to the output buffer
    size_t out_size = s1 + s2;
    memset(d_result, 0, out_size * sizeof(uint32_t));
    
    // Ensure proper alignment in the output
    size_t start_idx = (out_size > s_res) ? (out_size - s_res) : 0;
    for (size_t i = 0; i < s_res && (start_idx + i) < out_size; i++) {
        d_result[start_idx + i] = final[i];
    }
    
    free(final);
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

int main() {
    srand((unsigned int)time(NULL));

    for (long long i = 1; i <= 4294967296; i++) {  // Using 100 test cases for simplicity
        printf("========== TEST %lld ==========\n", i);
        
        char* num1 = gen(i);
        char* num2 = gen(i);
        if (!num1 || !num2) {
            fprintf(stderr, "Memory allocation failure\n");
            exit(EXIT_FAILURE);
        }

        if(strlen(num1) != strlen(num2)) {
            i--;
            free(num1);
            free(num2);
            continue;
        }
        
        // Calculate sizes (number of uint32_t words required)
        size_t s1 = ((strlen(num1) * 4) + 31) / 32;
        size_t s2 = ((strlen(num2) * 4) + 31) / 32;

        mpz_t g_num1, g_num2;
        mpz_init(g_num1);
        mpz_init(g_num2);
        mpz_set_str(g_num1, num1, 16);
        mpz_set_str(g_num2, num2, 16);

        uint32_t *h_num1 = (uint32_t*) malloc(s1 * sizeof(uint32_t));
        uint32_t *h_num2 = (uint32_t*) malloc(s2 * sizeof(uint32_t));
        uint32_t *h_result = (uint32_t*) malloc((s2+s1) * sizeof(uint32_t));
        if (!h_num1 || !h_num2 || !h_result) {
            fprintf(stderr, "Memory allocation failure\n");
            exit(EXIT_FAILURE);
        }

        mpz_t temp;
        mpz_init(temp);
        mpz_set(temp, g_num1);
        for (size_t j = 0; j < s1; j++) {
            h_num1[s1 - 1 - j] = (uint32_t) mpz_get_ui(temp);
            mpz_fdiv_q_2exp(temp, temp, 32);
        }        
        
        mpz_set(temp, g_num2);
        for (size_t j = 0; j < s2; j++) {
            h_num2[s2 - 1 - j] = (uint32_t) mpz_get_ui(temp);
            mpz_fdiv_q_2exp(temp, temp, 32);
        }
        mpz_clear(temp);

        kernel(h_num1, h_num2, s1, s2, h_result);

        char* hex_result = uint32_array_to_hex(h_result, (s1+s2));

        mpz_t g_mul;
        mpz_init(g_mul);
        mpz_mul(g_mul, g_num1, g_num2);
        char* gmp_hex_result = mpz_get_str(NULL, 16, g_mul);

        if (compare_strings(hex_result, gmp_hex_result)) {
            printf("The strings match (when compared from the end)!\n");
        } else {
            printf("The strings do not match (when compared from the end)!\n");
            FILE *file = fopen("output.txt", "a");
            if (file) {
                fprintf(file, "failed for %lld\n", i);
                fprintf(file, "num1: %s\n", num1);
                fprintf(file, "num2: %s\n", num2);
                fprintf(file, "\n");
                fclose(file);
            }
        }
        printf("\n");
        
        free(num1);
        free(num2);
        free(h_num1);
        free(h_num2);
        free(h_result);
        free(hex_result);
        free(gmp_hex_result);
        
        mpz_clear(g_mul);
        mpz_clear(g_num1);
        mpz_clear(g_num2);
    }
    return 0;
}