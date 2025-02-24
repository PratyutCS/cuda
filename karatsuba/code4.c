#include <iostream>
#include <string>
#include <vector>
#include <gmp.h>
#include <algorithm>
#include <cctype>
#include <fstream>

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
            // Instead of using addHex, we directly concatenate the two digits.
            char high = intToHexChar(product / 16);
            char low  = intToHexChar(product % 16);
            string res;
            res.push_back(high);
            res.push_back(low);
            return res;
        }
    }
    
    // Make both numbers the same length by left-padding with '0'
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

int main() {

    ifstream file("input-1000.txt");
    if (!file) {
        cerr << "Error: Could not open the file.\n";
        return 1;
    }

    vector<string> lines;
    string line;

    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }

    file.close();

    int row = 0;
    for (int i = lines.size(); i >= 1; i = ((i + 1)/ 2)) {
        row++;
        if (i == 1)
        break;
    }
    cout<<"rows required: "<<row<<endl;
    vector<vector<string>> aoa;
    aoa.push_back(lines);

    for(int i=1 ; i<row; i++){
        vector<string> temp;
        int size = (aoa[i-1].size()+1)/2;
        cout<<"size is : "<<size<<endl;
        for(int j=0 ; j<size ; j++){
            if(((j*2)+1) <= aoa[i-1].size()-1){
                temp.push_back(karatsubaHex(aoa[i-1][j*2], aoa[i-1][(j*2)+1]));
            }
            else{
                cout<<"else is true"<<endl;
                temp.push_back(aoa[i-1][j*2]);
            }
        }
        aoa.push_back(temp);
        temp.clear();
    }
    cout << "size of last is: " << aoa[row-1].size() << endl;
    for(int i=0 ; i<aoa[row-1].size() ; i++){
        cout<<aoa[row-1][i]<<endl;
    }

    // string hex1 = "8B141A795491F497B5C474389E880F1410DADA9C59CF70A514D3C1A99B299541061EDCC9939A4CF9AC76C348810D9DFE449F7771FCE0EB6D9B95D481576EE9B8D11E1BF11ED60CCA1BEC21C7D30525B8F4C1CF933A8C256CED1CE890E0AEF5E33205593E7B295B87568AFFF0DC56ACE9A960BEA08CE9ED67C495508435EF8A71F9DD44D96CACEF2849F6DF1DC5338FA4503621B880B1B43FC42730DEFC93691054A5E78A4EF35DC9ADB71C3929FB6F9418497D8DD2F65B22D3960ECA88851B76CAE73823A3064171A3395ED82419BE52F2C0DAA48CFD40D6B773DD97839EB219688F429F40A5D27EDD4581A11177CB48A9A96DE9E7088FC3CB7F7D1BC1CD4BC5";
    // string hex2 = "9910B4D2AD2E8F1D80729160B9FB0677C3ABBE52A69DE28238EA167EFD82FCFB628228104EDA0AF554FD1F85E71491FFA8EEA974A96127D7E81C09F301FBE95A14D2F83975F9119ED1C145B4EDA4AA7906B89071AD30E4022E02505280B1C966A61E5C4719164CEB7E09175D2B91CEACDA661E44D8CDA037F00CB0D9D382445B938F71F8DF14624FBAB726912B545EBAFA142335755FEC7D763104B53EFD4442EC4C23414FAF0FFFF88C2FA7CA897CE9BA7E9298A13242410AEC70CB9C6D6E64E422FD35AAFCF3BD3DB70DFA3AA2865BB93DA09502C3477E33CD53359B04032349ABAB5065E336A2939B17B09A6CAA41F8EA8100F6DC8C3471BDB72BEC7BA7D5";
    
    // // Perform Karatsuba multiplication
    // string karatsuba_result = karatsubaHex(hex1, hex2);
    // cout << "Karatsuba Product (Hexadecimal): " << karatsuba_result << endl;
    
    return 0;
}
