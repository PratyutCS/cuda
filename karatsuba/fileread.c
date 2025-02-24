#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main() {
    ifstream file("input-1000.txt");
    if (!file) {
        cerr << "Error: Could not open the file.\n";
        return 1;
    }

    vector<string> lines;
    string line;

    while (getline(file, line)) {
        lines.push_back(line);
    }

    file.close();

    for (const auto& l : lines) {
        cout << l << endl;
    }

    return 0;
}
