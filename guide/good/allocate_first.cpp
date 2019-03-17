#include <iostream>
#include <vector>

using namespace std;

struct Data {
    int data{0};
    Data(int data_) : data{data_} {}
};

void GenerateDataBad(vector<Data>& v, int n) {
    for (int i = 0; i < n; ++i) {
        v.push_back(Data{i});
    }
}

void GenerateDataGood(vector<Data>& v, int n) {
    v.reserve(n);  // with this line, it will way faster since no new allocation during push_back
    for (int i = 0; i < n; ++i) {
        v.push_back(Data{i});
    }
}

int main() {
    cout << "hello" << endl;
    return 0;
}
