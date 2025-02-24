#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <gmp.h>
#include <stdbool.h>
#include <ctype.h>
#include <cuda.h>

__device__ uint32_t* sum(uint32_t* a, uint32_t* b, size_t sa, size_t sb, size_t* res) {
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

__device__ uint32_t* diff(uint32_t* a, uint32_t* b, size_t sa, size_t sb, size_t* res) {
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

__device__ uint32_t* multiply(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, size_t* res){
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

    // free(p_num1);
    // free(p_num2);
    // free(a);
    // free(b);
    // free(c);
    // free(d);
    // free(ac);
    // free(bd);
    // free(a_plus_b);
    // free(c_plus_d);
    // free(a_plus_b_times_c_plus_d);
    // free(diff1);
    // free(diff2);
    // free(sum1);
    return sum2;
}

__global__ void kernel(uint32_t* num1, uint32_t* num2, size_t s1, size_t s2, uint32_t* d_result) {
    size_t s_res;
    uint32_t* final = multiply(num1, num2, s1, s2, &s_res);

    printf("final mul as hex (GPU):\n");
    for (size_t i = 0; i < s_res; i++) {
        printf("%08x ", final[i]);
    }
    printf("\n");

    printf("final mul as decimal (GPU):\n");
    for (size_t i = 0; i < s_res; i++) {
        printf("%u ",final[i]);
    }
    printf("\n");

    if(s_res > (s1+s2)){
        printf("greater than expected size diff of : %lu\n", (unsigned long)(s_res-(s1+s2)));
    }
    for(int i=0; i<(s1+s2); i++){
        d_result[(s1+s2) -1 -i] = (i<s_res) ? final[s_res-1-i] : 0 ;
    }

    return;
}

char* gen(int length) {
    char* hex_string = (char*)malloc((length + 1) * sizeof(char));
    if (hex_string == NULL) {
        return NULL;
    }
    
    const char hex_chars[] = "0123456789ABCDEF";
    for (int i = 0; i < length; i++) {
        int random_index = rand() % 16;
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

int main() {
    srand((unsigned int)time(NULL));

    for (int i = 95; i <= 95; i++) {
        printf("========== TEST %d ==========\n", i);
        // Generate host strings
        char* num1 = gen(i);
        char* num2 = gen(i);
        if (!num1 || !num2) {
            fprintf(stderr, "Memory allocation failure\n");
            exit(EXIT_FAILURE);
        }
        printf("num1: %s\n", num1);
        printf("num2: %s\n", num2);

        // Calculate sizes (number of uint32_t words required)
        size_t s1 = ((strlen(num1) * 4) + 31) / 32;
        size_t s2 = ((strlen(num2) * 4) + 31) / 32;

        mpz_t g_num1, g_num2;
        mpz_init(g_num1);
        mpz_init(g_num2);
        mpz_set_str(g_num1, num1, 16);
        gmp_printf("g_num1: %Zx\n", g_num1);
        mpz_set_str(g_num2, num2, 16);
        gmp_printf("g_num2: %Zx\n", g_num2);

        uint32_t *h_num1 = (uint32_t*) malloc(s1 * sizeof(uint32_t));
        uint32_t *h_num2 = (uint32_t*) malloc(s2 * sizeof(uint32_t));
        uint32_t *h_result = (uint32_t*) malloc((s2+s1) * sizeof(uint32_t));
        if (!h_num1 || !h_num2) {
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

        // Allocate device memory for the numbers
        uint32_t *d_num1, *d_num2, *d_result;
        cudaMalloc((void**)&d_num1, s1 * sizeof(uint32_t));
        cudaMalloc((void**)&d_num2, s2 * sizeof(uint32_t));
        cudaMalloc((void**)&d_result, (s1+s2) * sizeof(uint32_t));

        // Copy the numbers from host to device
        cudaMemcpy(d_num1, h_num1, s1 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num2, h_num2, s2 * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Launch the kernel by passing s1 and s2 directly
        kernel<<<1, 1>>>(d_num1, d_num2, s1, s2, d_result);
        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, (s1+s2) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Free the device memory
        cudaFree(d_num1);
        cudaFree(d_num2);
        cudaFree(d_result);

        // Free the host memory
        free(num1);
        free(num2);
        free(h_num1);
        free(h_num2);

        printf("final mul as hex (CPU):\n");
        for (size_t i = 0; i < (s1+s2); i++) {
            printf("%08x ", h_result[i]);
        }
        printf("\n");

        printf("final mul as decimal (CPU):\n");
        for (size_t i = 0; i < (s1+s2); i++) {
            printf("%u ",h_result[i]);
        }
        printf("\n");

        char* hex_result = uint32_array_to_hex(h_result, (s1+s2));
        printf("hexa converted in (cpu) is: %s\n",hex_result);

        mpz_t g_mul;
        mpz_init(g_mul);
        mpz_mul(g_mul, g_num1, g_num2);
        gmp_printf("mpz multiplucation result is: %Zx\n", g_mul);
        char* gmp_hex_result = mpz_get_str(NULL, 16, g_mul);

        if (compare_strings(hex_result, gmp_hex_result)) {
            printf("The strings match (when compared from the end)!\n");
        } else {
            printf("The strings do not match (when compared from the end)!\n");
        }
        printf("\n");
        
        mpz_clear(g_mul);
        mpz_clear(g_num1);
        mpz_clear(g_num2);
    }
    return 0;
}