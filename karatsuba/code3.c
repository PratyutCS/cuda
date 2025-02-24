#include <iostream>
#include <string>
#include <cstdlib>    // For rand, srand, malloc, free
#include <ctime>      // For time
#include <algorithm>  // For reverse
#include <cctype>     // For isdigit, toupper
#include <gmp.h>

using namespace std;

// Helper function: convert an integer (0-15) to its hexadecimal character.
char intToHexChar(int num) {
    return num < 10 ? num + '0' : num - 10 + 'A';
}

// Function to add two hexadecimal numbers (as strings)
string addHex(const string &a, const string &b) {
    string result;
    int carry = 0, sum;
    int i = a.size() - 1, j = b.size() - 1;
    
    while (i >= 0 || j >= 0 || carry) {
        sum = carry;
        if (i >= 0)
            sum += isdigit(a[i]) ? (a[i] - '0') : (toupper(a[i]) - 'A' + 10);
        if (j >= 0)
            sum += isdigit(b[j]) ? (b[j] - '0') : (toupper(b[j]) - 'A' + 10);
        carry = sum / 16;
        sum %= 16;
        result.push_back(intToHexChar(sum));
        i--, j--;
    }
    reverse(result.begin(), result.end());
    return result;
}

// Function to subtract two hexadecimal numbers (a - b)
string subtractHex(const string &a, const string &b) {
    string result;
    int borrow = 0, diff;
    int i = a.size() - 1, j = b.size() - 1;
    
    while (i >= 0) {
        diff = (isdigit(a[i]) ? (a[i] - '0') : (toupper(a[i]) - 'A' + 10)) - borrow;
        if (j >= 0)
            diff -= (isdigit(b[j]) ? (b[j] - '0') : (toupper(b[j]) - 'A' + 10));
        borrow = (diff < 0);
        if (borrow)
            diff += 16;
        result.push_back(intToHexChar(diff));
        i--, j--;
    }
    while (!result.empty() && result.back() == '0')
        result.pop_back();
    reverse(result.begin(), result.end());
    return result.empty() ? "0" : result;
}

// Function to multiply hex strings using the Karatsuba Algorithm
string karatsubaHex(const string &x, const string &y) {
    int n = max(x.size(), y.size());
    
    // Base case: single digit multiplication.
    if (n == 1) {
        int digit1 = isdigit(x[0]) ? x[0] - '0' : (toupper(x[0]) - 'A' + 10);
        int digit2 = isdigit(y[0]) ? y[0] - '0' : (toupper(y[0]) - 'A' + 10);
        int product = digit1 * digit2;
        if (product < 16) {
            return string(1, intToHexChar(product));
        } else {
            // Directly convert product to two hex digits.
            char high = intToHexChar(product / 16);
            char low  = intToHexChar(product % 16);
            string res;
            res.push_back(high);
            res.push_back(low);
            return res;
        }
    }
    
    // Pad the shorter string with leading zeros so both have length n.
    string X = x, Y = y;
    while (X.size() < n) X = "0" + X;
    while (Y.size() < n) Y = "0" + Y;
    
    int mid = n / 2;
    // Split the strings into two parts.
    string a = X.substr(0, X.size() - mid);
    string b = X.substr(X.size() - mid);
    string c = Y.substr(0, Y.size() - mid);
    string d = Y.substr(Y.size() - mid);
    
    // Recursively compute ac, bd, and (a+b)*(c+d)
    string ac = karatsubaHex(a, c);
    string bd = karatsubaHex(b, d);
    string aPlusB = addHex(a, b);
    string cPlusD = addHex(c, d);
    string ab_cd = karatsubaHex(aPlusB, cPlusD);
    string ad_bc = subtractHex(ab_cd, addHex(ac, bd));
    
    // Multiply ac by 16^(2*mid) (i.e., append 2*mid zeros) and ad_bc by 16^(mid)
    for (int i = 0; i < 2 * mid; i++) ac += "0";
    for (int i = 0; i < mid; i++) ad_bc += "0";
    
    // Return the sum: ac + (ad_bc) + bd
    return addHex(addHex(ac, ad_bc), bd);
}

// Function to generate a random hexadecimal string of specified length (C++ version)
string generateHexString(int length) {
    string hex;
    hex.reserve(length);
    const char hex_chars[] = "0123456789ABCDEF";
    for (int i = 0; i < length; i++) {
        int random_index = rand() % 16;
        hex.push_back(hex_chars[random_index]);
    }
    return hex;
}

int main() {
    // Seed the random number generator with the current time.
    srand(static_cast<unsigned int>(time(NULL)));
    
    // Loop through test cases with hex string lengths from 1 to 50.
    for (long long i = 1; i <= 4294967296; i++) {
        cout << "\n=== Test Case " << i << " (Hex strings of length " << i << ") ===" << endl;
        
        // Generate two random hexadecimal strings of length i.
        string hex1 = generateHexString(i);
        string hex2 = generateHexString(i);
        cout << "Hex 1: " << hex1 << endl;
        cout << "Hex 2: " << hex2 << endl;
        
        // Initialize GMP numbers for conversion.
        mpz_t num1, num2, mpz_result;
        mpz_init(num1);
        mpz_init(num2);
        mpz_init(mpz_result);
        if (mpz_set_str(num1, hex1.c_str(), 16) != 0 || mpz_set_str(num2, hex2.c_str(), 16) != 0) {
            cerr << "Invalid hexadecimal input" << endl;
            mpz_clear(num1);
            mpz_clear(num2);
            mpz_clear(mpz_result);
            continue;
        }
        
        // Perform Karatsuba multiplication.
        string karatsuba_result = karatsubaHex(hex1, hex2);
        cout << "Karatsuba Product (Hexadecimal): " << karatsuba_result << endl;
        
        // Convert the Karatsuba result to a GMP number.
        mpz_t karatsuba_product;
        mpz_init(karatsuba_product);
        if (mpz_set_str(karatsuba_product, karatsuba_result.c_str(), 16) != 0) {
            cerr << "Error in conversion of Karatsuba result" << endl;
            mpz_clear(karatsuba_product);
            mpz_clear(num1);
            mpz_clear(num2);
            mpz_clear(mpz_result);
            continue;
        }
        
        cout << "Karatsuba Product (Decimal): ";
        mpz_out_str(stdout, 10, karatsuba_product);
        cout << endl;
        
        // Perform GMP multiplication for comparison.
        mpz_mul(mpz_result, num1, num2);
        cout << "GMP Product (Hexadecimal): ";
        mpz_out_str(stdout, 16, mpz_result);
        cout << endl;
        cout << "GMP Product (Decimal): ";
        mpz_out_str(stdout, 10, mpz_result);
        cout << endl;
        
        // Compare the Karatsuba and GMP results using mpz_cmp.
        if (mpz_cmp(karatsuba_product, mpz_result) == 0) {
            cout << "Comparison: MATCH" << endl;
        } else {
            cout << "Comparison: MISMATCH" << endl;
            mpz_clear(num1);
            mpz_clear(num2);
            mpz_clear(mpz_result);
            mpz_clear(karatsuba_product);
            break;
        }
        
        // Free GMP memory for this test case.
        mpz_clear(num1);
        mpz_clear(num2);
        mpz_clear(mpz_result);
        mpz_clear(karatsuba_product);
    }
    
    return 0;
}
